// SPDX-License-Identifier: UNLICENSED

pragma solidity ^0.8.22;

import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";

interface IStaking {
    function transferStake(
        bytes32 coldkey,
        bytes32 hotkey,
        uint256 netuid1,
        uint256 netuid2,
        uint256 amount
    ) external payable;
    function moveStake(
        bytes32 hotkey1,
        bytes32 hotkey2,
        uint256 netuid1,
        uint256 netuid2,
        uint256 amount
    ) external payable;
    function getStake(
        bytes32 hotkey,
        bytes32 coldkey,
        uint256 netuid
    ) external view returns (uint256);
}

interface INeuron {
    function burnedRegister(uint16 netuid, bytes32 hotkey) external payable;
    function dummy() external payable;
}

contract CollateralUpgradeable is
    Initializable,
    UUPSUpgradeable,
    AccessControlUpgradeable
{
    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    // Version for tracking upgrades
    function getVersion() external pure virtual returns (uint256) {
        return 1;
    }

    // Role for upgrading the contract
    bytes32 public constant UPGRADER_ROLE = keccak256("UPGRADER_ROLE");

    address public constant ISTAKING_V2_ADDRESS =
        0x0000000000000000000000000000000000000805;

    address public constant INEURON_ADDRESS =
        0x0000000000000000000000000000000000000804;

    // State variables
    uint16 public NETUID;
    address public TRUSTEE;
    uint64 public DECISION_TIMEOUT;
    uint256 public MIN_COLLATERAL_INCREASE;
    bytes32 public CONTRACT_COLDKEY;
    bytes32 public CONTRACT_HOTKEY;

    mapping(bytes32 => mapping(bytes16 => address)) public nodeToMiner;
    mapping(bytes32 => mapping(bytes16 => uint256)) public collaterals;
    mapping(bytes32 => mapping(bytes16 => uint256)) public alphaCollaterals;
    mapping(uint256 => Reclaim) public reclaims;

    mapping(bytes32 => mapping(bytes16 => uint256))
        private collateralUnderPendingReclaims;
    mapping(bytes32 => mapping(bytes16 => uint256))
        private alphaCollateralUnderPendingReclaims;
    uint256 private nextReclaimId;

    struct Reclaim {
        bytes32 hotkey;
        bytes16 nodeId;
        address miner;
        uint256 amount;
        bytes32 alphaColdkey;
        uint256 alphaAmount;
        uint64 denyTimeout;
    }

    // Events
    event Deposit(
        bytes32 indexed hotkey,
        bytes16 indexed nodeId,
        address indexed miner,
        uint256 amount,
        bytes32 alphaHotkey,
        uint256 alphaAmount
    );
    event ReclaimProcessStarted(
        uint256 indexed reclaimRequestId,
        bytes32 indexed hotkey,
        bytes16 indexed nodeId,
        address miner,
        uint256 amount,
        bytes32 alphaColdkey,
        uint256 alphaAmount,
        uint64 expirationTime,
        string url,
        bytes16 urlContentMd5Checksum
    );
    event Reclaimed(
        uint256 indexed reclaimRequestId,
        bytes32 indexed hotkey,
        bytes16 indexed nodeId,
        address miner,
        uint256 amount,
        bytes32 alphaColdkey,
        uint256 alphaAmount
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
        uint256 slashAmount,
        uint256 slashAlphaAmount,
        string url,
        bytes16 urlContentMd5Checksum
    );
    event AlphaColdkeyUpdated(
        bytes32 indexed oldAlphaColdkey,
        bytes32 indexed newAlphaColdkey
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
    error InsufficientCollateralForSlash();
    error InvalidAlphaColdkey();

    /// @notice Initializes the upgradeable collateral contract
    /// @param netuid The netuid of the subnet
    /// @param trustee Address of the trustee who has permissions to slash collateral or deny reclaim requests
    /// @param minCollateralIncrease The minimum amount that can be deposited or reclaimed
    /// @param decisionTimeout The time window (in seconds) for the trustee to deny a reclaim request
    /// @param admin Address that will have admin and upgrader roles
    function initialize(
        uint16 netuid,
        address trustee,
        uint256 minCollateralIncrease,
        uint64 decisionTimeout,
        address admin,
        bytes32 alphaHotkey
    ) public initializer {
        require(trustee != address(0), "Trustee address must be non-zero");
        require(admin != address(0), "Admin address must be non-zero");
        require(alphaHotkey != bytes32(0), "Alpha hotkey must be non-zero");
        require(
            minCollateralIncrease > 0,
            "Min collateral increase must be greater than 0"
        );
        require(decisionTimeout > 0, "Decision timeout must be greater than 0");

        __UUPSUpgradeable_init();
        __AccessControl_init();

        NETUID = netuid;
        TRUSTEE = trustee;
        MIN_COLLATERAL_INCREASE = minCollateralIncrease;
        DECISION_TIMEOUT = decisionTimeout;
        CONTRACT_HOTKEY = alphaHotkey;

        // Set up roles
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(UPGRADER_ROLE, admin);
    }

    modifier onlyTrustee() {
        if (msg.sender != TRUSTEE) {
            revert NotTrustee();
        }
        _;
    }

    function setContractColdkey(bytes32 alphaColdkey) external onlyTrustee {
        require(alphaColdkey != bytes32(0), "Alpha coldkey must be non-zero");
        bytes32 oldAlphaColdkey = CONTRACT_COLDKEY;
        CONTRACT_COLDKEY = alphaColdkey;
        emit AlphaColdkeyUpdated(oldAlphaColdkey, alphaColdkey);
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
        bytes16 nodeId,
        bytes32 alphaHotkey,
        uint256 alphaAmount
    ) external payable {
        if (msg.value != 0 && msg.value < MIN_COLLATERAL_INCREASE) {
            revert InsufficientAmount();
        }

        address owner = nodeToMiner[hotkey][nodeId];
        if (owner == address(0)) {
            nodeToMiner[hotkey][nodeId] = msg.sender;
        } else if (owner != msg.sender) {
            revert NodeNotOwned();
        }

        uint256 actualAlphaAmount = alphaAmount;
        if (alphaAmount > 0) {
            require(
                CONTRACT_COLDKEY != bytes32(0),
                "contract coldkey must be non-zero"
            );
            actualAlphaAmount = transferAlpha(alphaHotkey, alphaAmount);
            alphaCollaterals[hotkey][nodeId] += actualAlphaAmount;
        }

        collaterals[hotkey][nodeId] += msg.value;

        emit Deposit(
            hotkey,
            nodeId,
            msg.sender,
            msg.value,
            alphaHotkey,
            actualAlphaAmount
        );
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
        bytes32 alphaColdkey,
        string calldata url,
        bytes16 urlContentMd5Checksum
    ) external {
        if (msg.sender != nodeToMiner[hotkey][nodeId]) {
            revert NodeNotOwned();
        }

        uint256 availableAmount = collaterals[hotkey][nodeId] -
            collateralUnderPendingReclaims[hotkey][nodeId];

        uint256 availableAlphaAmount = alphaCollaterals[hotkey][nodeId] -
            alphaCollateralUnderPendingReclaims[hotkey][nodeId];

        if (availableAmount == 0 && availableAlphaAmount == 0) {
            revert AmountZero();
        }

        if (availableAlphaAmount > 0 && alphaColdkey == bytes32(0)) {
            revert InvalidAlphaColdkey();
        }

        uint64 denyTimeout = uint64(block.timestamp) + DECISION_TIMEOUT;

        reclaims[nextReclaimId] = Reclaim({
            hotkey: hotkey,
            nodeId: nodeId,
            miner: msg.sender,
            amount: availableAmount,
            alphaColdkey: alphaColdkey,
            alphaAmount: availableAlphaAmount,
            denyTimeout: denyTimeout
        });

        collateralUnderPendingReclaims[hotkey][nodeId] += availableAmount;
        alphaCollateralUnderPendingReclaims[hotkey][
            nodeId
        ] += availableAlphaAmount;

        emit ReclaimProcessStarted(
            nextReclaimId,
            hotkey,
            nodeId,
            msg.sender,
            availableAmount,
            alphaColdkey,
            availableAlphaAmount,
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
        if (reclaim.amount == 0 && reclaim.alphaAmount == 0) {
            revert ReclaimNotFound();
        }
        if (reclaim.denyTimeout >= block.timestamp) {
            revert BeforeDenyTimeout();
        }

        bytes32 hotkey = reclaim.hotkey;
        bytes16 nodeId = reclaim.nodeId;
        address miner = reclaim.miner;
        uint256 amount = reclaim.amount;
        bytes32 alphaColdkey = reclaim.alphaColdkey;
        uint256 alphaAmount = reclaim.alphaAmount;

        delete reclaims[reclaimRequestId];
        collateralUnderPendingReclaims[hotkey][nodeId] -= amount;

        if (collaterals[hotkey][nodeId] < amount) {
            // miner got slashed and can't withdraw
            revert InsufficientCollateralForReclaim();
        }

        collaterals[hotkey][nodeId] -= amount;

        // check-effect-interact pattern used to prevent reentrancy attacks
        (bool success, ) = payable(miner).call{value: amount}("");
        if (!success) {
            revert TransferFailed();
        }

        if (alphaAmount > 0) {
            alphaCollaterals[hotkey][nodeId] -= alphaAmount;
            withdrawAlpha(alphaColdkey, alphaAmount);
        }

        if (
            collaterals[hotkey][nodeId] == 0 &&
            alphaCollaterals[hotkey][nodeId] == 0
        ) {
            nodeToMiner[hotkey][nodeId] = address(0);
        }

        emit Reclaimed(
            reclaimRequestId,
            hotkey,
            nodeId,
            miner,
            amount,
            alphaColdkey,
            alphaAmount
        );
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
        alphaCollateralUnderPendingReclaims[reclaim.hotkey][
            reclaim.nodeId
        ] -= reclaim.alphaAmount;
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
        uint256 slashAmount,
        uint256 slashAlphaAmount,
        string calldata url,
        bytes16 urlContentMd5Checksum
    ) external onlyTrustee {
        uint256 amount = collaterals[hotkey][nodeId];
        uint256 alphaAmount = alphaCollaterals[hotkey][nodeId];

        if (amount == 0 && alphaAmount == 0) {
            revert AmountZero();
        }

        if (slashAmount > amount || slashAlphaAmount > alphaAmount) {
            revert InsufficientCollateralForSlash();
        }

        collaterals[hotkey][nodeId] = amount - slashAmount;
        alphaCollaterals[hotkey][nodeId] = alphaAmount - slashAlphaAmount;
        address miner = nodeToMiner[hotkey][nodeId];

        // burn the collateral, alpha locked in the contract
        (bool success, ) = payable(address(0)).call{value: slashAmount}("");
        if (!success) {
            revert TransferFailed();
        }
        if (amount == slashAmount && alphaAmount == slashAlphaAmount) {
            nodeToMiner[hotkey][nodeId] = address(0);
        }
        emit Slashed(
            hotkey,
            nodeId,
            miner,
            slashAmount,
            slashAlphaAmount,
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

    function getContractStake(bytes32 hotkey) public view returns (uint256) {
        return
            IStaking(ISTAKING_V2_ADDRESS).getStake(
                hotkey,
                CONTRACT_COLDKEY,
                NETUID
            );
    }

    function transferAlpha(
        bytes32 alphaHotkey,
        uint256 alphaAmount
    ) internal returns (uint256) {
        uint256 contractStake = getContractStake(alphaHotkey);

        bytes memory data = abi.encodeWithSelector(
            IStaking.transferStake.selector,
            CONTRACT_COLDKEY,
            alphaHotkey,
            uint256(NETUID),
            uint256(NETUID),
            alphaAmount
        );
        // delegatecall the original sender should be used as origin for deposit alpha
        (bool success, ) = address(ISTAKING_V2_ADDRESS).delegatecall{
            gas: gasleft()
        }(data);
        require(success, "user deposit alpha call failed");

        uint256 newContractStake = getContractStake(alphaHotkey);

        require(
            newContractStake > contractStake,
            "contract stake decreased after deposit"
        );

        // use the increased stake as the actual alpha amount, for the swap fee in the move stake call
        // the contract will take it and get compensated by laster emission of alpha
        uint256 actualAlphaAmount = newContractStake - contractStake;

        if (alphaHotkey != CONTRACT_HOTKEY) {
            data = abi.encodeWithSelector(
                IStaking.moveStake.selector,
                alphaHotkey,
                CONTRACT_HOTKEY,
                NETUID,
                NETUID,
                actualAlphaAmount
            );
            // call the origin is the proxy contract. the alpha just transfer betweend different hotkeys of contract as coldkey
            (success, ) = address(ISTAKING_V2_ADDRESS).call{gas: gasleft()}(
                data
            );
            require(success, "user deposit, move stake call failed");
        }

        return actualAlphaAmount;
    }

    function withdrawAlpha(bytes32 alphaColdkey, uint256 alphaAmount) internal {
        uint256 contractStake = getContractStake(CONTRACT_HOTKEY);
        require(
            contractStake >= alphaAmount,
            "contract stake is less than withdraw alpha amount"
        );

        bytes memory data = abi.encodeWithSelector(
            IStaking.transferStake.selector,
            alphaColdkey,
            CONTRACT_HOTKEY,
            NETUID,
            NETUID,
            alphaAmount
        );
        // use call the origin should be the proxy contract
        (bool success, ) = address(ISTAKING_V2_ADDRESS).call{gas: gasleft()}(
            data
        );
        require(success, "user withdraw alpha call failed");
    }

    function burnRegister() external onlyTrustee {
        bytes memory data = abi.encodeWithSelector(
            INeuron.burnedRegister.selector,
            NETUID,
            CONTRACT_HOTKEY
        );
        (bool success, ) = address(INEURON_ADDRESS).call{gas: gasleft()}(data);
        require(success, "user burn register call failed");
    }
}
