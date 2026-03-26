// SPDX-License-Identifier: UNLICENSED
// The contract is the same as CollateralUpgradeable.sol, just different version number.
// Only for testing purposes.

pragma solidity ^0.8.22;

import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";

contract CollateralUpgradeableV2 is
    Initializable,
    UUPSUpgradeable,
    AccessControlUpgradeable
{
    constructor() {
        _disableInitializers();
    }

    // Version for tracking upgrades
    function getVersion() external pure virtual returns (uint256) {
        return 2;
    }

    // Role for upgrading the contract
    bytes32 public constant UPGRADER_ROLE = keccak256("UPGRADER_ROLE");

    // State variables
    uint16 public NETUID;
    address public TRUSTEE;
    uint64 public DECISION_TIMEOUT;
    uint256 public MIN_COLLATERAL_INCREASE;

    mapping(bytes32 => mapping(bytes16 => address)) public nodeToMiner;
    mapping(bytes32 => mapping(bytes16 => uint256)) public collaterals;
    mapping(uint256 => Reclaim) public reclaims;

    mapping(bytes32 => mapping(bytes16 => uint256))
        private collateralUnderPendingReclaims;
    uint256 private nextReclaimId;

    struct Reclaim {
        bytes32 hotkey;
        bytes16 nodeId;
        address miner;
        uint256 amount;
        uint64 denyTimeout;
    }

    // Events
    event Deposit(
        bytes32 indexed hotkey,
        bytes16 indexed nodeId,
        address indexed miner,
        uint256 amount
    );
    event ReclaimProcessStarted(
        uint256 indexed reclaimRequestId,
        bytes32 indexed hotkey,
        bytes16 indexed nodeId,
        address miner,
        uint256 amount,
        uint64 expirationTime,
        string url,
        bytes16 urlContentMd5Checksum
    );
    event Reclaimed(
        uint256 indexed reclaimRequestId,
        bytes32 indexed hotkey,
        bytes16 indexed nodeId,
        address miner,
        uint256 amount
    );
    event Denied(
        uint256 indexed reclaimRequestId,
        string url,
        bytes16 urlContentMd5Checksum
    );
    event Slashed(
        bytes32 indexed hotkey,
        bytes16 indexed nodeId,
        address indexed miner,
        uint256 amount,
        string url,
        bytes16 urlContentMd5Checksum
    );

    // Upgrade event
    event ContractUpgraded(
        uint256 indexed newVersion,
        address indexed newImplementation
    );

    // Custom errors
    error AmountZero();
    error BeforeDenyTimeout();
    error NodeNotOwned();
    error InsufficientAmount();
    error InvalidDepositMethod();
    error NotTrustee();
    error PastDenyTimeout();
    error ReclaimNotFound();
    error TransferFailed();
    error InsufficientCollateralForReclaim();

    function initializeV2() external {}

    modifier onlyTrustee() {
        if (msg.sender != TRUSTEE) {
            revert NotTrustee();
        }
        _;
    }

    // Allow deposits only via deposit() function
    receive() external payable {
        revert InvalidDepositMethod();
    }

    // Allow deposits only via deposit() function
    fallback() external payable {
        revert InvalidDepositMethod();
    }

    /// @notice Allows users to deposit collateral into the contract for a specific node
    /// @param hotkey The netuid key for the subnet
    /// @param nodeId The ID of the node to deposit collateral for
    /// @dev The first deposit for an nodeId sets the owner. Subsequent deposits must be from the owner.
    /// @dev The deposited amount must be greater than or equal to MIN_COLLATERAL_INCREASE
    /// @dev Emits a Deposit event with the hotkey, nodeId, sender's address and deposited amount
    function deposit(
        bytes32 hotkey,
        bytes16 nodeId
    ) external payable virtual {
        if (msg.value < MIN_COLLATERAL_INCREASE) {
            revert InsufficientAmount();
        }

        address owner = nodeToMiner[hotkey][nodeId];
        if (owner == address(0)) {
            nodeToMiner[hotkey][nodeId] = msg.sender;
        } else if (owner != msg.sender) {
            revert NodeNotOwned();
        }

        collaterals[hotkey][nodeId] += msg.value;
        emit Deposit(hotkey, nodeId, msg.sender, msg.value);
    }

    /// @notice Initiates a process to reclaim all available collateral from a specific node
    /// @dev If it's not denied by the trustee, the collateral will be available for withdrawal after DECISION_TIMEOUT
    /// @param hotkey The netuid key for the subnet
    /// @param nodeId The ID of the node to reclaim collateral from
    /// @param url URL containing information about the reclaim request
    /// @param urlContentMd5Checksum MD5 checksum of the content at the provided URL
    /// @dev Emits ReclaimProcessStarted event with reclaim details and timeout
    /// @dev Reverts with NodeNotOwned if caller is not the owner of the node
    /// @dev Reverts with AmountZero if there is no available collateral to reclaim
    function reclaimCollateral(
        bytes32 hotkey,
        bytes16 nodeId,
        string calldata url,
        bytes16 urlContentMd5Checksum
    ) external {
        if (msg.sender != nodeToMiner[hotkey][nodeId]) {
            revert NodeNotOwned();
        }

        uint256 totalCollateral = collaterals[hotkey][nodeId];
        uint256 pendingCollateral = collateralUnderPendingReclaims[hotkey][
            nodeId
        ];
        uint256 availableAmount = totalCollateral - pendingCollateral;

        if (availableAmount == 0) {
            revert AmountZero();
        }

        uint64 denyTimeout = uint64(block.timestamp) + DECISION_TIMEOUT;

        reclaims[nextReclaimId] = Reclaim({
            hotkey: hotkey,
            nodeId: nodeId,
            miner: msg.sender,
            amount: availableAmount,
            denyTimeout: denyTimeout
        });

        collateralUnderPendingReclaims[hotkey][nodeId] += availableAmount;

        emit ReclaimProcessStarted(
            nextReclaimId,
            hotkey,
            nodeId,
            msg.sender,
            availableAmount,
            denyTimeout,
            url,
            urlContentMd5Checksum
        );

        nextReclaimId++;
    }

    /// @notice Finalizes a reclaim request after the deny timeout has expired
    /// @dev Can only be called after the deny timeout has passed for the specific reclaim request
    /// @dev Transfers the collateral to the miner and removes the node-to-miner mapping if successful
    /// @dev This fully closes the relationship, allowing to request another reclaim
    /// @param reclaimRequestId The ID of the reclaim request to finalize
    /// @dev Emits Reclaimed event with reclaim details if successful
    /// @dev Reverts with ReclaimNotFound if the reclaim request doesn't exist or was denied
    /// @dev Reverts with BeforeDenyTimeout if the deny timeout hasn't expired
    /// @dev Reverts with TransferFailed if the TAO transfer fails
    function finalizeReclaim(uint256 reclaimRequestId) external {
        Reclaim storage reclaim = reclaims[reclaimRequestId];
        if (reclaim.amount == 0) {
            revert ReclaimNotFound();
        }
        if (reclaim.denyTimeout >= block.timestamp) {
            revert BeforeDenyTimeout();
        }

        bytes32 hotkey = reclaim.hotkey;
        bytes16 nodeId = reclaim.nodeId;
        address miner = reclaim.miner;
        uint256 amount = reclaim.amount;

        delete reclaims[reclaimRequestId];
        collateralUnderPendingReclaims[hotkey][nodeId] -= amount;

        if (collaterals[hotkey][nodeId] < amount) {
            // miner got slashed and can't withdraw
            revert InsufficientCollateralForReclaim();
        }

        collaterals[hotkey][nodeId] -= amount;

        emit Reclaimed(reclaimRequestId, hotkey, nodeId, miner, amount);

        // check-effect-interact pattern used to prevent reentrancy attacks
        (bool success, ) = payable(miner).call{value: amount}("");
        if (!success) {
            revert TransferFailed();
        }
        nodeToMiner[hotkey][nodeId] = address(0);
    }

    /// @notice Allows the trustee to deny a pending reclaim request before the timeout expires
    /// @dev Can only be called by the trustee (address set in initializer)
    /// @dev Must be called before the deny timeout expires
    /// @dev Removes the reclaim request and frees up the collateral for other reclaims
    /// @param reclaimRequestId The ID of the reclaim request to deny
    /// @param url URL containing the reason of denial
    /// @param urlContentMd5Checksum MD5 checksum of the content at the provided URL
    /// @dev Emits Denied event with the reclaim request ID
    /// @dev Reverts with NotTrustee if called by non-trustee address
    /// @dev Reverts with ReclaimNotFound if the reclaim request doesn't exist
    /// @dev Reverts with PastDenyTimeout if the timeout has already expired
    function denyReclaimRequest(
        uint256 reclaimRequestId,
        string calldata url,
        bytes16 urlContentMd5Checksum
    ) external onlyTrustee {
        Reclaim storage reclaim = reclaims[reclaimRequestId];
        if (reclaim.amount == 0) {
            revert ReclaimNotFound();
        }
        if (reclaim.denyTimeout < block.timestamp) {
            revert PastDenyTimeout();
        }

        collateralUnderPendingReclaims[reclaim.hotkey][
            reclaim.nodeId
        ] -= reclaim.amount;
        emit Denied(reclaimRequestId, url, urlContentMd5Checksum);

        delete reclaims[reclaimRequestId];
    }

    /// @notice Allows the trustee to slash a miner's collateral for a specific node
    /// @dev Can only be called by the trustee (address set in initializer)
    /// @dev Removes the collateral from the node and burns it
    /// @param hotkey The netuid key for the subnet
    /// @param nodeId The ID of the node to slash
    /// @param url URL containing the reason for slashing
    /// @param urlContentMd5Checksum MD5 checksum of the content at the provided URL
    /// @dev Emits Slashed event with the node's ID, miner's address and the amount slashed
    /// @dev Reverts with AmountZero if there is no collateral to slash
    /// @dev Reverts with TransferFailed if the TAO transfer fails
    function slashCollateral(
        bytes32 hotkey,
        bytes16 nodeId,
        string calldata url,
        bytes16 urlContentMd5Checksum
    ) external onlyTrustee {
        uint256 amount = collaterals[hotkey][nodeId];

        if (amount == 0) {
            revert AmountZero();
        }

        collaterals[hotkey][nodeId] = 0;
        address miner = nodeToMiner[hotkey][nodeId];

        // burn the collateral
        (bool success, ) = payable(address(0)).call{value: amount}("");
        if (!success) {
            revert TransferFailed();
        }
        nodeToMiner[hotkey][nodeId] = address(0);
        emit Slashed(
            hotkey,
            nodeId,
            miner,
            amount,
            url,
            urlContentMd5Checksum
        );
    }

    /// @notice Allows the trustee to partially slash a miner's collateral
    /// @dev Can only be called by the trustee (address set in initializer)
    /// @dev Burns a specified amount of collateral for the node
    /// @param hotkey The netuid key for the subnet
    /// @param nodeId The ID of the node to slash
    /// @param amount Amount of collateral to slash
    /// @param url URL containing the reason for slashing
    /// @param urlContentMd5Checksum MD5 checksum of the content at the provided URL
    /// @dev Emits Slashed event with the node's ID, miner's address and the amount slashed
    /// @dev Reverts with AmountZero if amount is zero
    /// @dev Reverts with InsufficientAmount if amount exceeds collateral
    /// @dev Reverts with TransferFailed if the TAO transfer fails
    function slashCollateralAmount(
        bytes32 hotkey,
        bytes16 nodeId,
        uint256 amount,
        string calldata url,
        bytes16 urlContentMd5Checksum
    ) external onlyTrustee {
        uint256 collateral = collaterals[hotkey][nodeId];

        if (amount == 0) {
            revert AmountZero();
        }
        if (collateral < amount) {
            revert InsufficientAmount();
        }

        collaterals[hotkey][nodeId] = collateral - amount;
        address miner = nodeToMiner[hotkey][nodeId];

        // burn the collateral
        (bool success, ) = payable(address(0)).call{value: amount}("");
        if (!success) {
            revert TransferFailed();
        }
        if (collaterals[hotkey][nodeId] == 0) {
            nodeToMiner[hotkey][nodeId] = address(0);
        }
        emit Slashed(
            hotkey,
            nodeId,
            miner,
            amount,
            url,
            urlContentMd5Checksum
        );
    }

    /// @notice Updates the trustee address
    /// @param newTrustee The new trustee address
    /// @dev Can only be called by accounts with DEFAULT_ADMIN_ROLE
    function updateTrustee(
        address newTrustee
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(newTrustee != address(0), "New trustee cannot be zero address");
        address oldTrustee = TRUSTEE;
        TRUSTEE = newTrustee;

        // Emit an event for the trustee change
        emit TrusteeUpdated(oldTrustee, newTrustee);
    }

    /// @notice Updates the decision timeout
    /// @param newTimeout The new decision timeout in seconds
    /// @dev Can only be called by accounts with DEFAULT_ADMIN_ROLE
    function updateDecisionTimeout(
        uint64 newTimeout
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(newTimeout > 0, "Decision timeout must be greater than 0");
        uint64 oldTimeout = DECISION_TIMEOUT;
        DECISION_TIMEOUT = newTimeout;

        // Emit an event for the timeout change
        emit DecisionTimeoutUpdated(oldTimeout, newTimeout);
    }

    /// @notice Updates the minimum collateral increase
    /// @param newMinIncrease The new minimum collateral increase
    /// @dev Can only be called by accounts with DEFAULT_ADMIN_ROLE
    function updateMinCollateralIncrease(
        uint256 newMinIncrease
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(
            newMinIncrease > 0,
            "Min collateral increase must be greater than 0"
        );
        uint256 oldMinIncrease = MIN_COLLATERAL_INCREASE;
        MIN_COLLATERAL_INCREASE = newMinIncrease;

        // Emit an event for the min increase change
        emit MinCollateralIncreaseUpdated(oldMinIncrease, newMinIncrease);
    }

    /// @dev Function to authorize upgrades, restricted to UPGRADER_ROLE
    function _authorizeUpgrade(
        address newImplementation
    ) internal override onlyRole(UPGRADER_ROLE) {
        emit ContractUpgraded(this.getVersion() + 1, newImplementation);
    }

    // Additional events for administrative changes
    event TrusteeUpdated(
        address indexed oldTrustee,
        address indexed newTrustee
    );
    event DecisionTimeoutUpdated(uint64 oldTimeout, uint64 newTimeout);
    event MinCollateralIncreaseUpdated(
        uint256 oldMinIncrease,
        uint256 newMinIncrease
    );
}
