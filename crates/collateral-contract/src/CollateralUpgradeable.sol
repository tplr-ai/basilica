// SPDX-License-Identifier: UNLICENSED

pragma solidity ^0.8.24;

import {Initializable} from "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import {UUPSUpgradeable} from "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import {AccessControlUpgradeable} from "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import {ReentrancyGuardUpgradeable} from "@openzeppelin/contracts-upgradeable/utils/ReentrancyGuardUpgradeable.sol";
import {Math} from "@openzeppelin/contracts/utils/math/Math.sol";
import {SafeCast} from "@openzeppelin/contracts/utils/math/SafeCast.sol";

interface IStaking {
    function transferStake(bytes32 coldkey, bytes32 hotkey, uint256 netuid1, uint256 netuid2, uint256 amount)
        external
        payable;
    function moveStake(bytes32 hotkey1, bytes32 hotkey2, uint256 netuid1, uint256 netuid2, uint256 amount)
        external
        payable;
    function getStake(bytes32 hotkey, bytes32 coldkey, uint256 netuid) external view returns (uint256);
}

interface INeuron {
    function burnedRegister(uint16 netuid, bytes32 hotkey) external payable;
    function dummy() external payable;
}

interface IAddressMapping {
    function addressMapping(address evmAddress) external view returns (bytes32);
}

contract CollateralUpgradeable is Initializable, UUPSUpgradeable, AccessControlUpgradeable, ReentrancyGuardUpgradeable {
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
    bytes32 public constant TRUSTEE_ROLE = keccak256("TRUSTEE_ROLE");

    address public constant ISTAKING_V2_ADDRESS = 0x0000000000000000000000000000000000000805;

    address public constant INEURON_ADDRESS = 0x0000000000000000000000000000000000000804;

    address public constant IADDRESS_MAPPING_ADDRESS = 0x000000000000000000000000000000000000080C;

    // State variables
    uint16 public netuid;
    address public trustee;
    uint64 public decisionTimeout;
    uint256 public minCollateralIncrease;
    bytes32 public contractColdkey;
    bytes32 public validatorHotkey;

    mapping(bytes32 => mapping(bytes16 => address)) public nodeToMiner;
    mapping(bytes32 => mapping(bytes16 => uint256)) public taoCollaterals;
    mapping(bytes32 => mapping(bytes16 => uint256)) public alphaCollaterals;
    mapping(uint256 => Reclaim) public reclaims;

    mapping(bytes32 => mapping(bytes16 => uint256)) private taoCollateralUnderPendingReclaims;
    mapping(bytes32 => mapping(bytes16 => uint256)) private alphaCollateralUnderPendingReclaims;
    uint256 private nextReclaimId;
    mapping(bytes32 => mapping(bytes16 => bytes32)) public ownerColdkeys;

    bool public taoDepositsEnabled;
    bool public alphaDepositsEnabled;

    /// @dev Reserved storage gap for future upgrades.
    uint256[49] private _gap;

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
        bytes32 urlContentSha256
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
    event Denied(uint256 indexed reclaimRequestId, string url, bytes32 urlContentSha256);
    event Slashed(
        bytes32 indexed hotkey,
        bytes16 indexed nodeId,
        address indexed miner,
        uint256 slashAmount,
        uint256 slashAlphaAmount,
        string url,
        bytes32 urlContentSha256
    );

    // Upgrade event
    event ContractUpgraded(uint256 indexed newVersion, address indexed newImplementation);

    // Custom errors
    error AmountZero();
    error BeforeDenyTimeout();
    error NodeNotOwned();
    error InsufficientAmount();
    error InvalidDepositMethod();
    error PastDenyTimeout();
    error ReclaimNotFound();
    error TransferFailed();
    error InsufficientCollateralForSlash();
    error InvalidAlphaColdkey();
    error TaoDepositsDisabled();
    error AlphaDepositsDisabled();
    error AddressMappingPrecompileCallFailed();
    error AddressMappingPrecompileInvalidResponse();
    error InvalidDerivedContractColdkey();
    error InvalidDerivedOwnerColdkey();
    error MinerMustBeEOA();
    error TrusteeAddressZero();
    error AdminAddressZero();
    error ValidatorHotkeyZero();
    error MinCollateralIncreaseZero();
    error DecisionTimeoutZero();
    error ContractColdkeyZero();
    error NewTrusteeAddressZero();
    error DepositAlphaCallFailed();
    error ContractStakeDidNotIncrease();
    error MoveStakeCallFailed();
    error ContractStakeTooLowForWithdraw();
    error WithdrawAlphaCallFailed();
    error BurnRegisterCallFailed();

    /// @notice Initializes the upgradeable collateral contract
    /// @param netuid_ The netuid of the subnet
    /// @param trustee_ Address of the trustee who has permissions to slash collateral or deny reclaim requests
    /// @param minCollateralIncrease_ The minimum TAO amount that can be deposited
    /// @param decisionTimeout_ The time window (in seconds) for the trustee to deny a reclaim request
    /// @param admin Address that will have admin and upgrader roles
    /// @param validatorHotkey_ The Substrate hotkey of the validator where all alpha is consolidated
    /// @param taoDepositsEnabled_ Whether TAO deposits are initially enabled
    /// @param alphaDepositsEnabled_ Whether alpha deposits are initially enabled
    function initialize(
        uint16 netuid_,
        address trustee_,
        uint256 minCollateralIncrease_,
        uint64 decisionTimeout_,
        address admin,
        bytes32 validatorHotkey_,
        bool taoDepositsEnabled_,
        bool alphaDepositsEnabled_
    ) public initializer {
        if (trustee_ == address(0)) {
            revert TrusteeAddressZero();
        }
        if (admin == address(0)) {
            revert AdminAddressZero();
        }
        if (validatorHotkey_ == bytes32(0)) {
            revert ValidatorHotkeyZero();
        }
        if (minCollateralIncrease_ == 0) {
            revert MinCollateralIncreaseZero();
        }
        if (decisionTimeout_ == 0) {
            revert DecisionTimeoutZero();
        }

        __UUPSUpgradeable_init();
        __AccessControl_init();
        __ReentrancyGuard_init();

        netuid = netuid_;
        trustee = trustee_;
        minCollateralIncrease = minCollateralIncrease_;
        decisionTimeout = decisionTimeout_;
        validatorHotkey = validatorHotkey_;
        contractColdkey = _deriveContractColdkey();
        taoDepositsEnabled = taoDepositsEnabled_;
        alphaDepositsEnabled = alphaDepositsEnabled_;

        // Set up roles
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(UPGRADER_ROLE, admin);
        _grantRole(TRUSTEE_ROLE, trustee_);
    }

    function _addressMapping(address evmAddress) internal view returns (bytes32) {
        (bool success, bytes memory returndata) =
            IADDRESS_MAPPING_ADDRESS.staticcall(abi.encodeCall(IAddressMapping.addressMapping, (evmAddress)));
        if (!success) {
            revert AddressMappingPrecompileCallFailed();
        }
        if (returndata.length != 32) {
            revert AddressMappingPrecompileInvalidResponse();
        }

        return abi.decode(returndata, (bytes32));
    }

    function _deriveContractColdkey() internal view returns (bytes32) {
        bytes32 derivedColdkey = _addressMapping(address(this));
        if (derivedColdkey == bytes32(0)) {
            revert InvalidDerivedContractColdkey();
        }

        return derivedColdkey;
    }

    function _deriveOwnerColdkey(address owner) internal view returns (bytes32) {
        bytes32 derivedColdkey = _addressMapping(owner);
        if (derivedColdkey == bytes32(0)) {
            revert InvalidDerivedOwnerColdkey();
        }
        return derivedColdkey;
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
    /// @param hotkey The miner's Bittensor hotkey under which the node is registered
    /// @param nodeId The ID of the node to deposit collateral for
    /// @param alphaHotkey The hotkey under which the miner's alpha is currently staked
    /// @param alphaAmount The amount of alpha to transfer as collateral (in RAO)
    /// @dev The first deposit for a nodeId sets the owner. Subsequent deposits must be from the owner.
    /// @dev The TAO deposit amount must be greater than or equal to minCollateralIncrease
    /// @dev Emits a Deposit event with the hotkey, nodeId, sender's address and deposited amount
    function deposit(bytes32 hotkey, bytes16 nodeId, bytes32 alphaHotkey, uint256 alphaAmount)
        external
        payable
        nonReentrant
    {
        if (msg.value == 0 && alphaAmount == 0) {
            revert AmountZero();
        }
        if (msg.value > 0 && !taoDepositsEnabled) {
            revert TaoDepositsDisabled();
        }
        if (alphaAmount > 0 && !alphaDepositsEnabled) {
            revert AlphaDepositsDisabled();
        }
        if (msg.value != 0 && msg.value < minCollateralIncrease) {
            revert InsufficientAmount();
        }

        address owner = nodeToMiner[hotkey][nodeId];
        if (owner == address(0)) {
            // Block constructor-based bypasses where code.length is zero during creation.
            if (msg.sender.code.length != 0 || tx.origin != msg.sender) {
                revert MinerMustBeEOA();
            }
            nodeToMiner[hotkey][nodeId] = msg.sender;
            ownerColdkeys[hotkey][nodeId] = _deriveOwnerColdkey(msg.sender);
        } else if (owner != msg.sender) {
            revert NodeNotOwned();
        } else if (ownerColdkeys[hotkey][nodeId] == bytes32(0)) {
            // Backfill owner coldkey for nodes that existed before this mapping.
            ownerColdkeys[hotkey][nodeId] = _deriveOwnerColdkey(msg.sender);
        }

        uint256 actualAlphaAmount = alphaAmount;
        if (alphaAmount > 0) {
            if (contractColdkey == bytes32(0)) {
                revert ContractColdkeyZero();
            }
            actualAlphaAmount = transferAlpha(alphaHotkey, alphaAmount);
            alphaCollaterals[hotkey][nodeId] += actualAlphaAmount;
        }

        taoCollaterals[hotkey][nodeId] += msg.value;

        emit Deposit(hotkey, nodeId, msg.sender, msg.value, alphaHotkey, actualAlphaAmount);
    }

    /// @notice Initiates a process to reclaim all available collateral from a specific node
    /// @dev If it's not denied by the trustee, the collateral will be available for withdrawal after decisionTimeout
    /// @param hotkey The miner's Bittensor hotkey under which the node is registered
    /// @param nodeId The ID of the node to reclaim collateral from
    /// @dev Alpha payout destination is always derived from the owner address mapping.
    /// @param url URL containing information about the reclaim request
    /// @param urlContentSha256 SHA-256 checksum of the content at the provided URL
    /// @dev Emits ReclaimProcessStarted event with reclaim details and timeout
    /// @dev Reverts with NodeNotOwned if caller is not the owner of the node
    /// @dev Reverts with AmountZero if there is no available collateral to reclaim
    function reclaimCollateral(bytes32 hotkey, bytes16 nodeId, string calldata url, bytes32 urlContentSha256)
        external
        nonReentrant
    {
        if (msg.sender != nodeToMiner[hotkey][nodeId]) {
            revert NodeNotOwned();
        }

        uint256 availableAmount =
            Math.saturatingSub(taoCollaterals[hotkey][nodeId], taoCollateralUnderPendingReclaims[hotkey][nodeId]);

        uint256 availableAlphaAmount =
            Math.saturatingSub(alphaCollaterals[hotkey][nodeId], alphaCollateralUnderPendingReclaims[hotkey][nodeId]);

        if (availableAmount == 0 && availableAlphaAmount == 0) {
            revert AmountZero();
        }

        bytes32 ownerColdkey = ownerColdkeys[hotkey][nodeId];
        if (ownerColdkey == bytes32(0)) {
            ownerColdkey = _deriveOwnerColdkey(msg.sender);
            ownerColdkeys[hotkey][nodeId] = ownerColdkey;
        }

        if (availableAlphaAmount > 0 && ownerColdkey == bytes32(0)) {
            revert InvalidAlphaColdkey();
        }

        uint64 denyTimeout = SafeCast.toUint64(block.timestamp + decisionTimeout);

        reclaims[nextReclaimId] = Reclaim({
            hotkey: hotkey,
            nodeId: nodeId,
            miner: msg.sender,
            amount: availableAmount,
            alphaColdkey: ownerColdkey,
            alphaAmount: availableAlphaAmount,
            denyTimeout: denyTimeout
        });

        taoCollateralUnderPendingReclaims[hotkey][nodeId] += availableAmount;
        alphaCollateralUnderPendingReclaims[hotkey][nodeId] += availableAlphaAmount;

        emit ReclaimProcessStarted(
            nextReclaimId,
            hotkey,
            nodeId,
            msg.sender,
            availableAmount,
            ownerColdkey,
            availableAlphaAmount,
            denyTimeout,
            url,
            urlContentSha256
        );

        nextReclaimId++;
    }

    /// @notice Finalizes a reclaim request after the deny timeout has expired
    /// @dev Can only be called after the deny timeout has passed for the specific reclaim request
    /// @dev Transfers the collateral to the miner. Clears the node-to-miner mapping only when all balances and pending reclaims reach zero
    /// @param reclaimRequestId The ID of the reclaim request to finalize
    /// @dev Emits Reclaimed event with reclaim details if successful
    /// @dev Reverts with ReclaimNotFound if the reclaim request doesn't exist or was denied
    /// @dev Reverts with BeforeDenyTimeout if the deny timeout hasn't expired
    /// @dev Reverts with TransferFailed if the TAO transfer fails
    function finalizeReclaim(uint256 reclaimRequestId) external nonReentrant {
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

        // --- Effects ---
        delete reclaims[reclaimRequestId];
        taoCollateralUnderPendingReclaims[hotkey][nodeId] -= amount;
        alphaCollateralUnderPendingReclaims[hotkey][nodeId] -= alphaAmount;

        // Cap TAO transfer to available balance (slash may have reduced it)
        uint256 actualAmount = Math.min(amount, taoCollaterals[hotkey][nodeId]);
        taoCollaterals[hotkey][nodeId] -= actualAmount;

        // Cap alpha transfer to available balance (slash may have reduced it)
        uint256 actualAlphaAmount = Math.min(alphaAmount, alphaCollaterals[hotkey][nodeId]);
        alphaCollaterals[hotkey][nodeId] -= actualAlphaAmount;

        if (
            taoCollaterals[hotkey][nodeId] == 0 && alphaCollaterals[hotkey][nodeId] == 0
                && taoCollateralUnderPendingReclaims[hotkey][nodeId] == 0
                && alphaCollateralUnderPendingReclaims[hotkey][nodeId] == 0
        ) {
            nodeToMiner[hotkey][nodeId] = address(0);
            ownerColdkeys[hotkey][nodeId] = bytes32(0);
        }

        emit Reclaimed(reclaimRequestId, hotkey, nodeId, miner, actualAmount, alphaColdkey, actualAlphaAmount);

        // --- Interactions ---
        if (actualAmount > 0) {
            (bool success,) = payable(miner).call{value: actualAmount}("");
            if (!success) {
                revert TransferFailed();
            }
        }

        if (actualAlphaAmount > 0) {
            withdrawAlpha(alphaColdkey, actualAlphaAmount);
        }
    }

    /// @notice Allows the trustee to deny a pending reclaim request before the timeout expires
    /// @dev Can only be called by an account with TRUSTEE_ROLE
    /// @dev Must be called before the deny timeout expires
    /// @dev Removes the reclaim request and frees up the collateral for other reclaims
    /// @param reclaimRequestId The ID of the reclaim request to deny
    /// @param url URL containing the reason of denial
    /// @param urlContentSha256 SHA-256 checksum of the content at the provided URL
    /// @dev Emits Denied event with the reclaim request ID
    /// @dev Reverts with AccessControlUnauthorizedAccount if called by non-trustee address
    /// @dev Reverts with ReclaimNotFound if the reclaim request doesn't exist
    /// @dev Reverts with PastDenyTimeout if the timeout has already expired
    function denyReclaimRequest(uint256 reclaimRequestId, string calldata url, bytes32 urlContentSha256)
        external
        onlyRole(TRUSTEE_ROLE)
        nonReentrant
    {
        Reclaim storage reclaim = reclaims[reclaimRequestId];
        if (reclaim.amount == 0 && reclaim.alphaAmount == 0) {
            revert ReclaimNotFound();
        }
        if (reclaim.denyTimeout < block.timestamp) {
            revert PastDenyTimeout();
        }

        bytes32 hotkey = reclaim.hotkey;
        bytes16 nodeId = reclaim.nodeId;

        taoCollateralUnderPendingReclaims[hotkey][nodeId] -= reclaim.amount;
        alphaCollateralUnderPendingReclaims[hotkey][nodeId] -= reclaim.alphaAmount;
        emit Denied(reclaimRequestId, url, urlContentSha256);

        delete reclaims[reclaimRequestId];

        // Clear ownership if all balances and pending reclaims are zero
        if (
            taoCollaterals[hotkey][nodeId] == 0 && alphaCollaterals[hotkey][nodeId] == 0
                && taoCollateralUnderPendingReclaims[hotkey][nodeId] == 0
                && alphaCollateralUnderPendingReclaims[hotkey][nodeId] == 0
        ) {
            nodeToMiner[hotkey][nodeId] = address(0);
            ownerColdkeys[hotkey][nodeId] = bytes32(0);
        }
    }

    /// @notice Allows the trustee to slash a miner's collateral for a specific node
    /// @dev Can only be called by an account with TRUSTEE_ROLE
    /// @dev Removes the collateral from the node and sends it to the trustee
    /// @param hotkey The miner's Bittensor hotkey under which the node is registered
    /// @param nodeId The ID of the node to slash
    /// @param slashAmount The amount of TAO collateral to slash (in wei)
    /// @param slashAlphaAmount The amount of alpha collateral to slash (in RAO)
    /// @param url URL containing the reason for slashing
    /// @param urlContentSha256 SHA-256 checksum of the content at the provided URL
    /// @dev Emits Slashed event with the node's ID, miner's address and the amount slashed
    /// @dev Reverts with AmountZero if there is no collateral to slash
    /// @dev Reverts with TransferFailed if the TAO transfer fails
    function slashCollateral(
        bytes32 hotkey,
        bytes16 nodeId,
        uint256 slashAmount,
        uint256 slashAlphaAmount,
        string calldata url,
        bytes32 urlContentSha256
    ) external onlyRole(TRUSTEE_ROLE) nonReentrant {
        uint256 amount = taoCollaterals[hotkey][nodeId];
        uint256 alphaAmount = alphaCollaterals[hotkey][nodeId];

        if (amount == 0 && alphaAmount == 0) {
            revert AmountZero();
        }

        if (slashAmount > amount || slashAlphaAmount > alphaAmount) {
            revert InsufficientCollateralForSlash();
        }

        taoCollaterals[hotkey][nodeId] = amount - slashAmount;
        alphaCollaterals[hotkey][nodeId] = alphaAmount - slashAlphaAmount;
        address miner = nodeToMiner[hotkey][nodeId];
        bytes32 trusteeColdkey = bytes32(0);
        if (slashAlphaAmount > 0) {
            trusteeColdkey = _deriveOwnerColdkey(msg.sender);
        }

        if (
            amount == slashAmount && alphaAmount == slashAlphaAmount
                && taoCollateralUnderPendingReclaims[hotkey][nodeId] == 0
                && alphaCollateralUnderPendingReclaims[hotkey][nodeId] == 0
        ) {
            nodeToMiner[hotkey][nodeId] = address(0);
            ownerColdkeys[hotkey][nodeId] = bytes32(0);
        }

        // send slashed TAO to the trustee
        if (slashAmount > 0) {
            (bool success,) = payable(trustee).call{value: slashAmount}("");
            if (!success) {
                revert TransferFailed();
            }
        }
        // slash alpha by transferring ownership to trustee coldkey
        if (slashAlphaAmount > 0) {
            withdrawAlpha(trusteeColdkey, slashAlphaAmount);
        }
        emit Slashed(hotkey, nodeId, miner, slashAmount, slashAlphaAmount, url, urlContentSha256);
    }

    /// @notice Updates the trustee address
    /// @param newTrustee The new trustee address
    /// @dev Can only be called by accounts with DEFAULT_ADMIN_ROLE
    function updateTrustee(address newTrustee) external onlyRole(DEFAULT_ADMIN_ROLE) {
        if (newTrustee == address(0)) {
            revert NewTrusteeAddressZero();
        }
        address oldTrustee = trustee;
        if (oldTrustee != newTrustee && oldTrustee != address(0)) {
            _revokeRole(TRUSTEE_ROLE, oldTrustee);
        }
        _grantRole(TRUSTEE_ROLE, newTrustee);
        trustee = newTrustee;

        // Emit an event for the trustee change
        emit TrusteeUpdated(oldTrustee, newTrustee);
    }

    /// @notice Updates the decision timeout
    /// @param newTimeout The new decision timeout in seconds
    /// @dev Can only be called by accounts with DEFAULT_ADMIN_ROLE
    function updateDecisionTimeout(uint64 newTimeout) external onlyRole(DEFAULT_ADMIN_ROLE) {
        if (newTimeout == 0) {
            revert DecisionTimeoutZero();
        }
        uint64 oldTimeout = decisionTimeout;
        decisionTimeout = newTimeout;

        // Emit an event for the timeout change
        emit DecisionTimeoutUpdated(oldTimeout, newTimeout);
    }

    /// @notice Updates the minimum collateral increase
    /// @param newMinIncrease The new minimum collateral increase
    /// @dev Can only be called by accounts with DEFAULT_ADMIN_ROLE
    function updateMinCollateralIncrease(uint256 newMinIncrease) external onlyRole(DEFAULT_ADMIN_ROLE) {
        if (newMinIncrease == 0) {
            revert MinCollateralIncreaseZero();
        }
        uint256 oldMinIncrease = minCollateralIncrease;
        minCollateralIncrease = newMinIncrease;

        // Emit an event for the min increase change
        emit MinCollateralIncreaseUpdated(oldMinIncrease, newMinIncrease);
    }

    function updateTaoDepositsEnabled(bool enabled) external onlyRole(DEFAULT_ADMIN_ROLE) {
        taoDepositsEnabled = enabled;
        emit TaoDepositsEnabledUpdated(enabled);
    }

    function updateAlphaDepositsEnabled(bool enabled) external onlyRole(DEFAULT_ADMIN_ROLE) {
        alphaDepositsEnabled = enabled;
        emit AlphaDepositsEnabledUpdated(enabled);
    }

    /// @dev Function to authorize upgrades, restricted to UPGRADER_ROLE
    function _authorizeUpgrade(address newImplementation) internal override onlyRole(UPGRADER_ROLE) {
        emit ContractUpgraded(this.getVersion() + 1, newImplementation);
    }

    // Additional events for administrative changes
    event TrusteeUpdated(address indexed oldTrustee, address indexed newTrustee);
    event DecisionTimeoutUpdated(uint64 oldTimeout, uint64 newTimeout);
    event MinCollateralIncreaseUpdated(uint256 oldMinIncrease, uint256 newMinIncrease);
    event TaoDepositsEnabledUpdated(bool enabled);
    event AlphaDepositsEnabledUpdated(bool enabled);

    function getContractStake(bytes32 hotkey) public view returns (uint256) {
        return IStaking(ISTAKING_V2_ADDRESS).getStake(hotkey, contractColdkey, netuid);
    }

    function transferAlpha(bytes32 alphaHotkey, uint256 alphaAmount) internal returns (uint256) {
        uint256 contractStake = getContractStake(alphaHotkey);

        bytes memory data = abi.encodeWithSelector(
            IStaking.transferStake.selector, contractColdkey, alphaHotkey, uint256(netuid), uint256(netuid), alphaAmount
        );
        // delegatecall the original sender should be used as origin for deposit alpha
        (bool success,) = address(ISTAKING_V2_ADDRESS).delegatecall{gas: gasleft()}(data);
        if (!success) {
            revert DepositAlphaCallFailed();
        }

        uint256 newContractStake = getContractStake(alphaHotkey);

        if (newContractStake <= contractStake) {
            revert ContractStakeDidNotIncrease();
        }

        // use the increased stake as the actual alpha amount, for the swap fee in the move stake call
        // the contract will take it and get compensated by later emission of alpha
        uint256 actualAlphaAmount = newContractStake - contractStake;

        if (alphaHotkey != validatorHotkey) {
            data = abi.encodeWithSelector(
                IStaking.moveStake.selector, alphaHotkey, validatorHotkey, netuid, netuid, actualAlphaAmount
            );
            // call the origin is the proxy contract. the alpha just transfers between different hotkeys of contract as coldkey
            (success,) = address(ISTAKING_V2_ADDRESS).call{gas: gasleft()}(data);
            if (!success) {
                revert MoveStakeCallFailed();
            }
        }

        return actualAlphaAmount;
    }

    function withdrawAlpha(bytes32 alphaColdkey, uint256 alphaAmount) internal {
        uint256 contractStake = getContractStake(validatorHotkey);
        if (contractStake < alphaAmount) {
            revert ContractStakeTooLowForWithdraw();
        }

        bytes memory data = abi.encodeWithSelector(
            IStaking.transferStake.selector, alphaColdkey, validatorHotkey, netuid, netuid, alphaAmount
        );
        // use call the origin should be the proxy contract
        (bool success,) = address(ISTAKING_V2_ADDRESS).call{gas: gasleft()}(data);
        if (!success) {
            revert WithdrawAlphaCallFailed();
        }
    }

    function burnRegister() external onlyRole(TRUSTEE_ROLE) {
        bytes memory data = abi.encodeWithSelector(INeuron.burnedRegister.selector, netuid, validatorHotkey);
        (bool success,) = address(INEURON_ADDRESS).call{gas: gasleft()}(data);
        if (!success) {
            revert BurnRegisterCallFailed();
        }
    }
}
