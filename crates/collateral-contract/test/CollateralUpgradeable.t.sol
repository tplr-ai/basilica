// SPDX-License-Identifier: UNLICENSED

pragma solidity ^0.8.24;

import {Test} from "forge-std/Test.sol";
import {CollateralUpgradeable} from "../src/CollateralUpgradeable.sol";
import {CollateralUpgradeableUpgradeMock} from "./mocks/CollateralUpgradeableUpgradeMock.sol";
import {ERC1967Proxy} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";

contract AddressMappingPrecompileMock {
    function addressMapping(address evmAddress) external pure returns (bytes32) {
        return bytes32(uint256(uint160(evmAddress)));
    }
}

contract StakingV2PrecompileMock {
    function transferStake(
        bytes32,
        bytes32,
        uint256,
        uint256,
        uint256
    ) external payable {}

    function moveStake(
        bytes32,
        bytes32,
        uint256,
        uint256,
        uint256
    ) external payable {}

    function getStake(bytes32, bytes32, uint256) external view returns (uint256) {
        return type(uint256).max - gasleft();
    }
}

contract ContractDepositorUpgradeable {
    function claimNode(
        CollateralUpgradeable collateral,
        bytes32 hotkey,
        bytes16 nodeId,
        bytes32 alphaHotkey,
        uint256 alphaAmount
    ) external payable {
        collateral.deposit{value: msg.value}(
            hotkey,
            nodeId,
            alphaHotkey,
            alphaAmount
        );
    }
}

contract CollateralUpgradeableTest is Test {
    CollateralUpgradeable public collateral;
    CollateralUpgradeable public implementation;
    ERC1967Proxy public proxy;

    // Test parameters
    uint16 constant NETUID = 39;
    address constant TRUSTEE = address(0x123);
    uint256 constant MIN_DEPOSIT = 1 ether;
    uint64 constant DECISION_TIMEOUT = 3600; // 1 hour
    address constant ADMIN = address(0x456);
    address constant ALICE = address(0x789);
    bytes32 constant ALPHA_HOTKEY = bytes32(uint256(1));
    uint256 constant ALPHA_AMOUNT = 1 ether;
    address constant ADDRESS_MAPPING_PRECOMPILE =
        0x000000000000000000000000000000000000080C;
    address constant STAKING_V2_PRECOMPILE =
        0x0000000000000000000000000000000000000805;

    function setUp() public {
        AddressMappingPrecompileMock addressMappingMock = new AddressMappingPrecompileMock();
        vm.etch(ADDRESS_MAPPING_PRECOMPILE, address(addressMappingMock).code);
        StakingV2PrecompileMock stakingMock = new StakingV2PrecompileMock();
        vm.etch(STAKING_V2_PRECOMPILE, address(stakingMock).code);

        // Deploy implementation
        implementation = new CollateralUpgradeable();

        // Prepare initialization data
        bytes memory initData = abi.encodeWithSelector(
            CollateralUpgradeable.initialize.selector,
            NETUID,
            TRUSTEE,
            MIN_DEPOSIT,
            DECISION_TIMEOUT,
            ADMIN,
            ALPHA_HOTKEY
        );

        // Deploy proxy
        proxy = new ERC1967Proxy(address(implementation), initData);

        // Cast proxy to interface
        collateral = CollateralUpgradeable(payable(address(proxy)));
    }

    function _nodeToMinerSlot(
        bytes32 hotkey,
        bytes16 nodeId
    ) internal pure returns (bytes32) {
        uint256 nodeToMinerSlot = 4;
        bytes32 levelOne = keccak256(abi.encode(hotkey, nodeToMinerSlot));
        return keccak256(abi.encode(nodeId, levelOne));
    }

    function _seedNodeOwner(
        bytes32 hotkey,
        bytes16 nodeId,
        address owner
    ) internal {
        vm.store(
            address(collateral),
            _nodeToMinerSlot(hotkey, nodeId),
            bytes32(uint256(uint160(owner)))
        );
    }

    /// @dev Test basic initialization
    function testInitialization() public view {
        assertEq(collateral.netuid(), NETUID);
        assertEq(collateral.trustee(), TRUSTEE);
        assertEq(collateral.minCollateralIncrease(), MIN_DEPOSIT);
        assertEq(collateral.decisionTimeout(), DECISION_TIMEOUT);
        assertEq(collateral.getVersion(), 1);

        // Check roles
        assertTrue(collateral.hasRole(collateral.DEFAULT_ADMIN_ROLE(), ADMIN));
        assertTrue(collateral.hasRole(collateral.UPGRADER_ROLE(), ADMIN));
    }

    /// @dev Test that implementation cannot be initialized directly
    function testImplementationCannotBeInitialized() public {
        CollateralUpgradeable directImplementation = new CollateralUpgradeable();

        vm.expectRevert(); // Should revert due to _disableInitializers()
        directImplementation.initialize(
            NETUID,
            TRUSTEE,
            MIN_DEPOSIT,
            DECISION_TIMEOUT,
            ADMIN,
            ALPHA_HOTKEY
        );
    }

    /// @dev Test basic deposit functionality
    function testBasicDeposit() public {
        vm.deal(ALICE, 10 ether);
        bytes32 hotkey = bytes32(uint256(1));
        bytes16 nodeId = bytes16(uint128(1));
        _seedNodeOwner(hotkey, nodeId, ALICE);

        // Test event emission
        vm.expectEmit(true, true, true, true, address(collateral));
        emit Deposit(hotkey, nodeId, ALICE, 5 ether, ALPHA_HOTKEY, 0);

        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(hotkey, nodeId, ALPHA_HOTKEY, 0);
        // Verify state
        assertEq(collateral.taoCollaterals(hotkey, nodeId), 5 ether);
        assertEq(collateral.nodeToMiner(hotkey, nodeId), ALICE);
        assertEq(address(collateral).balance, 5 ether);
    }

    function testFirstDepositRequiresAlphaForOwnershipClaim() public {
        vm.deal(ALICE, 10 ether);
        bytes32 hotkey = bytes32(uint256(0xA1));
        bytes16 nodeId = bytes16(uint128(0xB1));

        vm.prank(ALICE);
        vm.expectRevert(
            abi.encodeWithSelector(
                CollateralUpgradeable.AlphaRequiredForOwnership.selector
            )
        );
        collateral.deposit{value: 5 ether}(hotkey, nodeId, ALPHA_HOTKEY, 0);
    }

    function testAlphaOwnershipClaimThenTaoTopUp() public {
        vm.deal(ALICE, 10 ether);
        bytes32 hotkey = bytes32(uint256(0xA2));
        bytes16 nodeId = bytes16(uint128(0xB2));

        vm.prank(ALICE);
        collateral.deposit(hotkey, nodeId, ALPHA_HOTKEY, 1);
        assertEq(collateral.nodeToMiner(hotkey, nodeId), ALICE);
        assertEq(collateral.taoCollaterals(hotkey, nodeId), 0);
        assertGt(collateral.alphaCollaterals(hotkey, nodeId), 0);

        vm.prank(ALICE);
        collateral.deposit{value: 2 ether}(hotkey, nodeId, ALPHA_HOTKEY, 0);
        assertEq(collateral.taoCollaterals(hotkey, nodeId), 2 ether);
    }

    function testFirstOwnershipClaimMustBeEOA() public {
        ContractDepositorUpgradeable depositor = new ContractDepositorUpgradeable();
        bytes32 hotkey = bytes32(uint256(0xA3));
        bytes16 nodeId = bytes16(uint128(0xB3));

        vm.expectRevert(
            abi.encodeWithSelector(CollateralUpgradeable.MinerMustBeEOA.selector)
        );
        depositor.claimNode(collateral, hotkey, nodeId, ALPHA_HOTKEY, 1);
    }

    /// @dev Test ADMIN functions
    function testAdminFunctions() public {
        address newTrustee = makeAddr("newTrustee");

        // Test TRUSTEE update
        vm.expectEmit(true, true, false, false, address(collateral));
        emit TrusteeUpdated(TRUSTEE, newTrustee);

        vm.prank(ADMIN);
        collateral.updateTrustee(newTrustee);
        assertEq(collateral.trustee(), newTrustee);

        // Test decision timeout update
        vm.prank(ADMIN);
        collateral.updateDecisionTimeout(7200); // 2 hours
        assertEq(collateral.decisionTimeout(), 7200);

        // Test min collateral increase update
        vm.prank(ADMIN);
        collateral.updateMinCollateralIncrease(2 ether);
        assertEq(collateral.minCollateralIncrease(), 2 ether);
    }

    /// @dev Test contract upgrade functionality
    function testUpgrade() public {
        // Deploy new implementation
        CollateralUpgradeableUpgradeMock newImplementation = new CollateralUpgradeableUpgradeMock();

        // Test event emission
        vm.expectEmit(true, true, false, false, address(collateral));
        emit ContractUpgraded(2, address(newImplementation));

        // Upgrade to new implementation
        vm.prank(ADMIN);
        collateral.upgradeToAndCall(address(newImplementation), "");

        // Verify upgrade
        assertEq(collateral.getVersion(), 2);

        // Verify state is preserved
        assertEq(collateral.netuid(), NETUID);
        assertEq(collateral.trustee(), TRUSTEE);
        assertEq(collateral.decisionTimeout(), DECISION_TIMEOUT);
        assertEq(collateral.minCollateralIncrease(), MIN_DEPOSIT);

        // Verify ADMIN still has roles
        assertTrue(collateral.hasRole(collateral.DEFAULT_ADMIN_ROLE(), ADMIN));
        assertTrue(collateral.hasRole(collateral.UPGRADER_ROLE(), ADMIN));
    }

    function testUpgradePreservesFlow() public {
        vm.deal(ALICE, 10 ether);
        bytes32 hotkey = bytes32(uint256(0xA4));
        bytes16 nodeId = bytes16(uint128(0xB4));

        vm.prank(ALICE);
        collateral.deposit(hotkey, nodeId, ALPHA_HOTKEY, 1);
        vm.prank(ALICE);
        collateral.deposit{value: 3 ether}(hotkey, nodeId, ALPHA_HOTKEY, 0);

        CollateralUpgradeableUpgradeMock newImplementation = new CollateralUpgradeableUpgradeMock();
        vm.prank(ADMIN);
        collateral.upgradeToAndCall(address(newImplementation), "");
        assertEq(collateral.getVersion(), 2);

        vm.prank(ALICE);
        collateral.reclaimCollateral(
            hotkey,
            nodeId,
            "https://example.com/reclaim",
            bytes32(uint256(1))
        );

        vm.warp(block.timestamp + DECISION_TIMEOUT + 1);
        uint256 aliceBalanceBefore = ALICE.balance;
        collateral.finalizeReclaim(0);

        assertEq(ALICE.balance, aliceBalanceBefore + 3 ether);
        assertEq(collateral.taoCollaterals(hotkey, nodeId), 0);
    }

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
    event Denied(
        uint256 indexed reclaimRequestId,
        string url,
        bytes32 urlContentSha256
    );
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
    event ContractUpgraded(
        uint256 indexed newVersion,
        address indexed newImplementation
    );

    event TrusteeUpdated(
        address indexed oldTrustee,
        address indexed newTrustee
    );

    event DecisionTimeoutUpdated(uint64 oldTimeout, uint64 newTimeout);

    event MinCollateralIncreaseUpdated(
        uint256 oldMinIncrease,
        uint256 newMinIncrease
    );

    /// @dev Test that denyReclaimRequest clears nodeToMiner after full slash
    function testDenyAfterFullSlashClearsNodeToMiner() public {
        vm.deal(ALICE, 10 ether);
        bytes32 hotkey = bytes32(uint256(1));
        bytes16 nodeId = bytes16(uint128(1));
        _seedNodeOwner(hotkey, nodeId, ALICE);

        // 1. Alice deposits
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(hotkey, nodeId, ALPHA_HOTKEY, 0);
        assertEq(collateral.nodeToMiner(hotkey, nodeId), ALICE);

        // 2. Alice starts a reclaim
        vm.prank(ALICE);
        collateral.reclaimCollateral(
            hotkey,
            nodeId,
            "https://example.com",
            bytes16(0)
        );

        // 3. Trustee fully slashes (TAO collateral goes to zero, but pending reclaim keeps nodeToMiner)
        vm.prank(TRUSTEE);
        collateral.slashCollateral(
            hotkey,
            nodeId,
            5 ether,
            0,
            "https://example.com/slash",
            bytes16(0)
        );
        // nodeToMiner should still be set because of pending reclaim
        assertEq(collateral.nodeToMiner(hotkey, nodeId), ALICE);

        // 4. Trustee denies the reclaim — should now clear nodeToMiner
        vm.prank(TRUSTEE);
        collateral.denyReclaimRequest(
            0,
            "https://example.com/deny",
            bytes16(0)
        );
        assertEq(
            collateral.nodeToMiner(hotkey, nodeId),
            address(0),
            "nodeToMiner should be cleared after deny with zero balances"
        );

        // 5. New miner (bob) can deposit on the same node
        address bob = address(0xB0B);
        vm.deal(bob, 10 ether);
        vm.prank(bob);
        collateral.deposit{value: 2 ether}(hotkey, nodeId, ALPHA_HOTKEY, 1);
        assertEq(collateral.nodeToMiner(hotkey, nodeId), bob);
    }
}
