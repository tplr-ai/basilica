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
    ) external;
    function moveStake(
        bytes32 hotkey1,
        bytes32 hotkey2,
        uint256 netuid1,
        uint256 netuid2,
        uint256 amount
    ) external;
    function getStake(
        bytes32 hotkey,
        bytes32 coldkey,
        uint256 netuid
    ) external view returns (uint256);
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

    // State variables
    uint16 public NETUID;
    address public TRUSTEE;
    uint64 public DECISION_TIMEOUT;
    uint256 public MIN_COLLATERAL_INCREASE;
    bytes32 public CONTRACT_COLDKEY;
    bytes32 public CONTRACT_HOTKEY;

    mapping(bytes32 => mapping(bytes16 => address)) public executorToMiner;
    mapping(bytes32 => mapping(bytes16 => uint256)) public collaterals;
    mapping(bytes32 => mapping(bytes16 => uint256)) internal alphaCollaterals;
    mapping(uint256 => Reclaim) public reclaims;

    mapping(bytes32 => mapping(bytes16 => uint256))
        private collateralUnderPendingReclaims;
    mapping(bytes32 => mapping(bytes16 => uint256))
        private alphaCollateralUnderPendingReclaims;
    uint256 private nextReclaimId;

    struct Reclaim {
        bytes32 hotkey;
        bytes16 executorId;
        address miner;
        uint256 amount;
        bytes32 alphaColdkey;
        uint256 alphaAmount;
        uint64 denyTimeout;
    }

    // Events
    event Deposit(
        bytes32 indexed hotkey,
        bytes16 indexed executorId,
        address indexed miner,
        uint256 amount,
        bytes32 alphaHotkey,
        uint256 alphaAmount
    );
    event ReclaimProcessStarted(
        uint256 indexed reclaimRequestId,
        bytes32 indexed hotkey,
        bytes16 indexed executorId,
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
        bytes16 indexed executorId,
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
        bytes16 indexed executorId,
        address indexed miner,
        uint256 slashAmount,
        uint256 slashAlphaAmount,
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
    error ExecutorNotOwned();
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
        CONTRACT_COLDKEY = alphaColdkey;
    }

    // Allow deposits only via deposit() function
    receive() external payable {
        revert InvalidDepositMethod();
    }

    // Allow deposits only via deposit() function
    fallback() external payable {
        revert InvalidDepositMethod();
    }

    /// @notice Allows users to deposit collateral into the contract for a specific executor
    /// @param hotkey The netuid key for the subnet
    /// @param executorId The ID of the executor to deposit collateral for
    /// @dev The first deposit for an executorId sets the owner. Subsequent deposits must be from the owner.
    /// @dev The deposited amount must be greater than or equal to MIN_COLLATERAL_INCREASE
    /// @dev Emits a Deposit event with the hotkey, executorId, sender's address and deposited amount
    function deposit(
        bytes32 hotkey,
        bytes16 executorId,
        bytes32 alphaHotkey,
        uint256 alphaAmount
    ) external payable virtual {
        if (msg.value != 0 && msg.value < MIN_COLLATERAL_INCREASE) {
            revert InsufficientAmount();
        }

        address owner = executorToMiner[hotkey][executorId];
        if (owner == address(0)) {
            executorToMiner[hotkey][executorId] = msg.sender;
        } else if (owner != msg.sender) {
            revert ExecutorNotOwned();
        }

        uint256 actualAlphaAmount = transferAlpha(alphaHotkey, alphaAmount);

        collaterals[hotkey][executorId] += msg.value;
        alphaCollaterals[hotkey][executorId] += actualAlphaAmount;

        emit Deposit(
            hotkey,
            executorId,
            msg.sender,
            msg.value,
            alphaHotkey,
            actualAlphaAmount
        );
    }

    /// @notice Initiates a process to reclaim all available collateral from a specific executor
    /// @dev If it's not denied by the trustee, the collateral will be available for withdrawal after DECISION_TIMEOUT
    /// @param hotkey The netuid key for the subnet
    /// @param executorId The ID of the executor to reclaim collateral from
    /// @param url URL containing information about the reclaim request
    /// @param urlContentMd5Checksum MD5 checksum of the content at the provided URL
    /// @dev Emits ReclaimProcessStarted event with reclaim details and timeout
    /// @dev Reverts with ExecutorNotOwned if caller is not the owner of the executor
    /// @dev Reverts with AmountZero if there is no available collateral to reclaim
    function reclaimCollateral(
        bytes32 hotkey,
        bytes16 executorId,
        bytes32 alphaColdkey,
        string calldata url,
        bytes16 urlContentMd5Checksum
    ) external {
        if (msg.sender != executorToMiner[hotkey][executorId]) {
            revert ExecutorNotOwned();
        }

        uint256 totalCollateral = collaterals[hotkey][executorId];
        uint256 pendingCollateral = collateralUnderPendingReclaims[hotkey][
            executorId
        ];
        uint256 availableAmount = totalCollateral - pendingCollateral;

        uint256 alphaCollateral = alphaCollaterals[hotkey][executorId];
        uint256 pendingAlphaCollateral = alphaCollateralUnderPendingReclaims[
            hotkey
        ][executorId];
        uint256 availableAlphaAmount = alphaCollateral - pendingAlphaCollateral;

        if (availableAmount == 0 && availableAlphaAmount == 0) {
            revert AmountZero();
        }

        if (availableAlphaAmount > 0 && alphaColdkey == bytes32(0)) {
            revert InvalidAlphaColdkey();
        }

        uint64 denyTimeout = uint64(block.timestamp) + DECISION_TIMEOUT;

        reclaims[nextReclaimId] = Reclaim({
            hotkey: hotkey,
            executorId: executorId,
            miner: msg.sender,
            amount: availableAmount,
            alphaColdkey: alphaColdkey,
            alphaAmount: availableAlphaAmount,
            denyTimeout: denyTimeout
        });

        collateralUnderPendingReclaims[hotkey][executorId] += availableAmount;
        alphaCollateralUnderPendingReclaims[hotkey][
            executorId
        ] += availableAlphaAmount;

        emit ReclaimProcessStarted(
            nextReclaimId,
            hotkey,
            executorId,
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
    /// @dev Transfers the collateral to the miner and removes the executor-to-miner mapping if successful
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
        bytes16 executorId = reclaim.executorId;
        address miner = reclaim.miner;
        uint256 amount = reclaim.amount;

        delete reclaims[reclaimRequestId];
        collateralUnderPendingReclaims[hotkey][executorId] -= amount;

        if (collaterals[hotkey][executorId] < amount) {
            // miner got slashed and can't withdraw
            revert InsufficientCollateralForReclaim();
        }

        collaterals[hotkey][executorId] -= amount;

        // check-effect-interact pattern used to prevent reentrancy attacks
        (bool success, ) = payable(miner).call{value: amount}("");
        if (!success) {
            revert TransferFailed();
        }

        if (reclaim.alphaAmount > 0) {
            withdrawAlpha(reclaim.alphaColdkey, reclaim.alphaAmount);
        }

        if (collaterals[hotkey][executorId] == 0 && reclaim.alphaAmount == 0) {
            executorToMiner[hotkey][executorId] = address(0);
        }

        emit Reclaimed(
            reclaimRequestId,
            hotkey,
            executorId,
            miner,
            amount,
            reclaim.alphaColdkey,
            reclaim.alphaAmount
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
            reclaim.executorId
        ] -= reclaim.amount;
        alphaCollateralUnderPendingReclaims[reclaim.hotkey][
            reclaim.executorId
        ] -= reclaim.alphaAmount;
        emit Denied(reclaimRequestId, url, urlContentMd5Checksum);

        delete reclaims[reclaimRequestId];
    }

    /// @notice Allows the trustee to slash a miner's collateral for a specific executor
    /// @dev Can only be called by the trustee (address set in initializer)
    /// @dev Removes the collateral from the executor and burns it
    /// @param hotkey The netuid key for the subnet
    /// @param executorId The ID of the executor to slash
    /// @param url URL containing the reason for slashing
    /// @param urlContentMd5Checksum MD5 checksum of the content at the provided URL
    /// @dev Emits Slashed event with the executor's ID, miner's address and the amount slashed
    /// @dev Reverts with AmountZero if there is no collateral to slash
    /// @dev Reverts with TransferFailed if the TAO transfer fails
    function slashCollateral(
        bytes32 hotkey,
        bytes16 executorId,
        uint256 slashAmount,
        uint256 slashAlphaAmount,
        string calldata url,
        bytes16 urlContentMd5Checksum
    ) external onlyTrustee {
        uint256 amount = collaterals[hotkey][executorId];
        uint256 alphaAmount = alphaCollaterals[hotkey][executorId];

        if (amount == 0 && alphaAmount == 0) {
            revert AmountZero();
        }

        if (slashAmount > amount && slashAlphaAmount > alphaAmount) {
            revert InsufficientCollateralForSlash();
        }

        collaterals[hotkey][executorId] = amount - slashAmount;
        alphaCollaterals[hotkey][executorId] = alphaAmount - slashAlphaAmount;
        address miner = executorToMiner[hotkey][executorId];

        // burn the collateral, alpha locked in the contract
        (bool success, ) = payable(address(0)).call{value: slashAmount}("");
        if (!success) {
            revert TransferFailed();
        }
        if (amount == slashAmount && alphaAmount == slashAlphaAmount) {
            executorToMiner[hotkey][executorId] = address(0);
        }
        emit Slashed(
            hotkey,
            executorId,
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

    function getContractStake() public view returns (uint256) {
        return
            IStaking(ISTAKING_V2_ADDRESS).getStake(
                CONTRACT_HOTKEY,
                CONTRACT_COLDKEY,
                NETUID
            );
    }

    function transferAlpha(
        bytes32 alphaHotkey,
        uint256 alphaAmount
    ) internal returns (uint256) {
        uint256 contractStake = getContractStake();

        bytes memory data = abi.encodeWithSelector(
            IStaking.transferStake.selector,
            CONTRACT_COLDKEY,
            alphaHotkey,
            NETUID,
            NETUID,
            alphaAmount
        );
        (bool success, ) = address(ISTAKING_V2_ADDRESS).delegatecall{
            gas: gasleft()
        }(data);
        require(success, "user deposit alpha call failed");

        uint256 newContractStake = getContractStake();

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
            (success, ) = address(ISTAKING_V2_ADDRESS).call{gas: gasleft()}(
                data
            );
            require(success, "user deposit, move stake call failed");
        }

        return actualAlphaAmount;
    }

    function withdrawAlpha(bytes32 alphaColdkey, uint256 alphaAmount) internal {
        uint256 contractStake = getContractStake();
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
        (bool success, ) = address(ISTAKING_V2_ADDRESS).delegatecall{
            gas: gasleft()
        }(data);
        require(success, "user withdraw alpha call failed");
    }
}
