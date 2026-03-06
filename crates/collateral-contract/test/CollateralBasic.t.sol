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
    function transferStake(bytes32, bytes32, uint256, uint256, uint256) external payable {}

    function moveStake(bytes32, bytes32, uint256, uint256, uint256) external payable {}

    function getStake(bytes32, bytes32, uint256) external view returns (uint256) {
        return type(uint256).max - gasleft();
    }
}

contract ContractDepositor {
    function claimNode(
        CollateralUpgradeable collateral,
        bytes32 hotkey,
        bytes16 nodeId,
        bytes32 alphaHotkey,
        uint256 alphaAmount
    ) external payable {
        collateral.deposit{value: msg.value}(hotkey, nodeId, alphaHotkey, alphaAmount);
    }
}

/**
 * @title CollateralBasicTest
 * @notice Basic tests for CollateralUpgradeable without alpha/IStaking interactions
 * @dev These tests focus on core functionality that doesn't require IStaking mocking
 */
contract CollateralBasicTest is Test {
    CollateralUpgradeable public collateral;
    CollateralUpgradeable public implementation;
    ERC1967Proxy public proxy;

    // Test parameters
    uint16 constant NETUID = 42;
    address constant TRUSTEE = address(0x123);
    uint256 constant MIN_DEPOSIT = 1 ether;
    uint64 constant DECISION_TIMEOUT = 3600; // 1 hour
    address constant ADMIN = address(0x456);
    address constant ALICE = address(0x789);
    address constant BOB = address(0xABC);
    address constant CHARLIE = address(0xDEF);
    address constant ADDRESS_MAPPING_PRECOMPILE = 0x000000000000000000000000000000000000080C;
    address constant STAKING_V2_PRECOMPILE = 0x0000000000000000000000000000000000000805;

    bytes32 constant ALPHA_HOTKEY = bytes32(uint256(1));
    bytes32 constant HOTKEY_1 = bytes32(uint256(100));
    bytes32 constant HOTKEY_2 = bytes32(uint256(101));
    bytes16 constant EXECUTOR_ID_1 = bytes16(uint128(1));
    bytes16 constant EXECUTOR_ID_2 = bytes16(uint128(2));

    string constant TEST_URL = "https://example.com/proof";
    bytes32 constant TEST_SHA256 = bytes32(0x1234567890123456789012345678901212345678901234567890123456789012);

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
            MIN_DEPOSIT,
            DECISION_TIMEOUT,
            ADMIN,
            ALPHA_HOTKEY,
            true,
            true
        );

        // Deploy proxy
        proxy = new ERC1967Proxy(address(implementation), initData);

        // Cast proxy to interface
        collateral = CollateralUpgradeable(payable(address(proxy)));

        // Most legacy TAO flow tests operate on this node pair after ownership is established.
        _seedNodeOwner(HOTKEY_1, EXECUTOR_ID_1, ALICE);

        // Give test accounts some ETH
        vm.deal(ALICE, 100 ether);
        vm.deal(BOB, 100 ether);
        vm.deal(CHARLIE, 100 ether);
    }

    function _nodeToMinerSlot(bytes32 hotkey, bytes16 nodeId) internal pure returns (bytes32) {
        uint256 nodeToMinerSlot = 5;
        bytes32 levelOne = keccak256(abi.encode(hotkey, nodeToMinerSlot));
        return keccak256(abi.encode(nodeId, levelOne));
    }

    function _seedNodeOwner(bytes32 hotkey, bytes16 nodeId, address owner) internal {
        vm.store(address(collateral), _nodeToMinerSlot(hotkey, nodeId), bytes32(uint256(uint160(owner))));
    }

    function _mappedColdkey(address evmAddress) internal pure returns (bytes32) {
        return bytes32(uint256(uint160(evmAddress)));
    }

    function _deployCollateralWithDepositToggles(bool taoEnabled, bool alphaEnabled)
        internal
        returns (CollateralUpgradeable collateralWithToggles)
    {
        bytes memory initData = abi.encodeWithSelector(
            CollateralUpgradeable.initialize.selector,
            NETUID,
            TRUSTEE,
            MIN_DEPOSIT,
            MIN_DEPOSIT,
            DECISION_TIMEOUT,
            ADMIN,
            ALPHA_HOTKEY,
            taoEnabled,
            alphaEnabled
        );
        ERC1967Proxy localProxy = new ERC1967Proxy(address(implementation), initData);
        collateralWithToggles = CollateralUpgradeable(payable(address(localProxy)));
    }

    function _allCollaterals() internal view returns (CollateralUpgradeable.NodeCollateral[] memory) {
        return collateral.getAllCollaterals(0, type(uint256).max);
    }

    function _allReclaims() internal view returns (CollateralUpgradeable.ReclaimInfo[] memory) {
        return collateral.getAllReclaims(0, type(uint256).max);
    }

    // ============ INITIALIZATION TESTS ============

    function testInitialization() public view {
        assertEq(collateral.netuid(), NETUID);
        assertEq(collateral.trustee(), TRUSTEE);
        assertEq(collateral.minCollateralIncrease(), MIN_DEPOSIT);
        assertEq(collateral.decisionTimeout(), DECISION_TIMEOUT);
        assertEq(collateral.getVersion(), 1);
        assertEq(collateral.contractColdkey(), bytes32(uint256(uint160(address(proxy)))));
        assertEq(collateral.validatorHotkey(), ALPHA_HOTKEY);

        // Check roles
        assertTrue(collateral.hasRole(collateral.DEFAULT_ADMIN_ROLE(), ADMIN));
        assertTrue(collateral.hasRole(collateral.UPGRADER_ROLE(), ADMIN));
        assertTrue(collateral.hasRole(collateral.TRUSTEE_ROLE(), TRUSTEE));
    }

    function testCannotInitializeTwice() public {
        vm.expectRevert();
        collateral.initialize(
            NETUID, TRUSTEE, MIN_DEPOSIT, MIN_DEPOSIT, DECISION_TIMEOUT, ADMIN, ALPHA_HOTKEY, true, true
        );
    }

    // ============ DEPOSIT TESTS (WITHOUT ALPHA) ============

    function testDepositInsufficientAmount() public {
        vm.prank(ALICE);
        vm.expectRevert(abi.encodeWithSelector(CollateralUpgradeable.InsufficientAmount.selector));
        collateral.deposit{value: 0.5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);
    }

    function testDepositExecutorNotOwned() public {
        // Alice makes first deposit
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        // Bob tries to deposit to same executor
        vm.prank(BOB);
        vm.expectRevert(abi.encodeWithSelector(CollateralUpgradeable.NodeNotOwned.selector));
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);
    }

    function testMultipleDepositsFromSameOwner() public {
        // First deposit
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        // Second deposit from same owner
        vm.prank(ALICE);
        collateral.deposit{value: 3 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        assertEq(collateral.taoCollaterals(HOTKEY_1, EXECUTOR_ID_1), 8 ether);
        assertEq(collateral.nodeToMiner(HOTKEY_1, EXECUTOR_ID_1), ALICE);
    }

    function testTaoOnlyFirstDepositSucceeds() public {
        vm.prank(ALICE, ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_2, EXECUTOR_ID_2, ALPHA_HOTKEY, 0);
        assertEq(collateral.nodeToMiner(HOTKEY_2, EXECUTOR_ID_2), ALICE);
        assertEq(collateral.taoCollaterals(HOTKEY_2, EXECUTOR_ID_2), 5 ether);
    }

    function testFirstOwnershipClaimMustBeEOA() public {
        ContractDepositor depositor = new ContractDepositor();

        vm.expectRevert(abi.encodeWithSelector(CollateralUpgradeable.MinerMustBeEOA.selector));
        depositor.claimNode(collateral, HOTKEY_2, EXECUTOR_ID_2, ALPHA_HOTKEY, 1 ether);
    }

    function testAlphaClaimAllowsSubsequentTaoTopUp() public {
        vm.prank(ALICE, ALICE);
        collateral.deposit(HOTKEY_2, EXECUTOR_ID_2, ALPHA_HOTKEY, 1 ether);

        assertEq(collateral.nodeToMiner(HOTKEY_2, EXECUTOR_ID_2), ALICE);
        assertEq(collateral.taoCollaterals(HOTKEY_2, EXECUTOR_ID_2), 0);
        assertGt(collateral.alphaCollaterals(HOTKEY_2, EXECUTOR_ID_2), 0);

        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_2, EXECUTOR_ID_2, ALPHA_HOTKEY, 0);
        assertEq(collateral.taoCollaterals(HOTKEY_2, EXECUTOR_ID_2), 5 ether);
    }

    // ============ RECLAIM TESTS ============

    function testReclaimCollateral() public {
        // Setup: Alice deposits
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        // Alice initiates reclaim
        vm.expectEmit(true, true, true, true, address(collateral));
        emit ReclaimProcessStarted(
            0,
            HOTKEY_1,
            EXECUTOR_ID_1,
            ALICE,
            5 ether,
            _mappedColdkey(ALICE),
            0,
            uint64(block.timestamp + DECISION_TIMEOUT)
        );

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        // Check reclaim was created
        (bytes32 hotkey, bytes16 executorId, address miner, uint256 amount, bytes32 alphaColdkey,, uint64 denyTimeout) =
            collateral.reclaims(0);
        assertEq(hotkey, HOTKEY_1);
        assertEq(executorId, EXECUTOR_ID_1);
        assertEq(miner, ALICE);
        assertEq(amount, 5 ether);
        assertEq(alphaColdkey, _mappedColdkey(ALICE));
        assertEq(denyTimeout, block.timestamp + DECISION_TIMEOUT);
    }

    function testReclaimExecutorNotOwned() public {
        // Alice deposits
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        // Bob tries to reclaim
        vm.prank(BOB);
        vm.expectRevert(abi.encodeWithSelector(CollateralUpgradeable.NodeNotOwned.selector));
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);
    }

    function testReclaimAmountZero() public {
        // Try to reclaim without any deposits
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        vm.prank(ALICE);
        vm.expectRevert(abi.encodeWithSelector(CollateralUpgradeable.AmountZero.selector));
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);
    }

    function testReclaimAfterFullSlashReturnsAmountZeroNotUnderflow() public {
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        vm.prank(TRUSTEE);
        collateral.slashCollateral(HOTKEY_1, EXECUTOR_ID_1, 5 ether, 0, TEST_URL, TEST_SHA256);

        vm.prank(ALICE);
        vm.expectRevert(abi.encodeWithSelector(CollateralUpgradeable.AmountZero.selector));
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);
    }

    // ============ FINALIZE RECLAIM TESTS ============

    function testFinalizeReclaim() public {
        // Setup: Alice deposits and initiates reclaim
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        // Fast forward past timeout
        vm.warp(block.timestamp + DECISION_TIMEOUT + 1);

        uint256 aliceBalanceBefore = ALICE.balance;

        // Finalize reclaim
        vm.expectEmit(true, true, true, true, address(collateral));
        emit Reclaimed(0, HOTKEY_1, EXECUTOR_ID_1, ALICE, 5 ether, _mappedColdkey(ALICE), 0);

        collateral.finalizeReclaim(0);

        // Check state
        assertEq(ALICE.balance, aliceBalanceBefore + 5 ether);
        assertEq(collateral.taoCollaterals(HOTKEY_1, EXECUTOR_ID_1), 0);
        assertEq(collateral.nodeToMiner(HOTKEY_1, EXECUTOR_ID_1), address(0));
    }

    function testFinalizeReclaimBeforeTimeout() public {
        // Setup reclaim
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        // Try to finalize before timeout
        vm.expectRevert(abi.encodeWithSelector(CollateralUpgradeable.BeforeDenyTimeout.selector));
        collateral.finalizeReclaim(0);
    }

    function testFinalizeReclaimAtExactTimeoutSucceeds() public {
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        uint64 denyTimeout;
        (,,,,,, denyTimeout) = collateral.reclaims(0);
        vm.warp(uint256(denyTimeout));

        uint256 aliceBalanceBefore = ALICE.balance;
        collateral.finalizeReclaim(0);

        assertEq(ALICE.balance, aliceBalanceBefore + 5 ether);
        assertEq(collateral.taoCollaterals(HOTKEY_1, EXECUTOR_ID_1), 0);
        assertEq(collateral.nodeToMiner(HOTKEY_1, EXECUTOR_ID_1), address(0));
    }

    function testFinalizeReclaimNotFound() public {
        vm.expectRevert(abi.encodeWithSelector(CollateralUpgradeable.ReclaimNotFound.selector));
        collateral.finalizeReclaim(999);
    }

    function testFinalizeReclaimPartialAfterSlash() public {
        // Setup reclaim
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        // Slash 3 ETH of collateral during pending reclaim
        vm.prank(TRUSTEE);
        collateral.slashCollateral(HOTKEY_1, EXECUTOR_ID_1, 3 ether, 0, TEST_URL, TEST_SHA256);

        // Fast forward past timeout
        vm.warp(block.timestamp + DECISION_TIMEOUT + 1);

        uint256 aliceBalanceBefore = ALICE.balance;

        // Expect Reclaimed event with actual amount (2 ETH, not 5)
        vm.expectEmit(true, true, true, true, address(collateral));
        emit Reclaimed(0, HOTKEY_1, EXECUTOR_ID_1, ALICE, 2 ether, _mappedColdkey(ALICE), 0);

        // Finalize should succeed with partial amount
        collateral.finalizeReclaim(0);

        // Alice receives only 2 ETH (5 deposited - 3 slashed)
        assertEq(ALICE.balance, aliceBalanceBefore + 2 ether);
        assertEq(collateral.taoCollaterals(HOTKEY_1, EXECUTOR_ID_1), 0);
        assertEq(collateral.nodeToMiner(HOTKEY_1, EXECUTOR_ID_1), address(0));
    }

    function testFinalizeReclaimAfterFullSlash() public {
        // Setup reclaim
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        // Slash ALL collateral during pending reclaim
        vm.prank(TRUSTEE);
        collateral.slashCollateral(HOTKEY_1, EXECUTOR_ID_1, 5 ether, 0, TEST_URL, TEST_SHA256);

        // Fast forward past timeout
        vm.warp(block.timestamp + DECISION_TIMEOUT + 1);

        uint256 aliceBalanceBefore = ALICE.balance;

        // Expect Reclaimed event with 0 amounts
        vm.expectEmit(true, true, true, true, address(collateral));
        emit Reclaimed(0, HOTKEY_1, EXECUTOR_ID_1, ALICE, 0, _mappedColdkey(ALICE), 0);

        // Finalize should succeed even with 0 transfer
        collateral.finalizeReclaim(0);

        // Alice receives nothing
        assertEq(ALICE.balance, aliceBalanceBefore);
        assertEq(collateral.taoCollaterals(HOTKEY_1, EXECUTOR_ID_1), 0);
        // Reclaim is cleaned up
        (,,, uint256 amount,,,) = collateral.reclaims(0);
        assertEq(amount, 0);
    }

    function testDenyAlphaOnlyReclaim() public {
        // Setup: Alice deposits TAO only (so she owns the node)
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        // Create a normal TAO reclaim to get reclaimId 0
        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        // Verify reclaim 0 exists with amount=5 ether
        (,,, uint256 amt,, uint256 alphaAmt,) = collateral.reclaims(0);
        assertEq(amt, 5 ether);
        assertEq(alphaAmt, 0);

        // Compute storage slots for reclaim struct fields using vm.store
        // OZ v5 uses ERC-7201 namespaced storage, so our state vars start at slot 0:
        // slot 0: NETUID(u16) + TRUSTEE(addr) + DECISION_TIMEOUT(u64) packed
        // slot 1: MIN_COLLATERAL_INCREASE(u256)
        // slot 2: MIN_ALPHA_COLLATERAL_INCREASE(u256)
        // slot 3: CONTRACT_COLDKEY(b32)
        // slot 4: VALIDATOR_HOTKEY(b32)
        // slot 5: nodeToMiner mapping
        // slot 6: taoCollaterals mapping
        // slot 7: alphaCollaterals mapping
        // slot 8: reclaims mapping
        // slot 9: taoCollateralUnderPendingReclaims mapping
        // slot 10: alphaCollateralUnderPendingReclaims mapping
        // slot 11: nextReclaimId
        // slot 12: ownerColdkeys mapping
        uint256 reclaimsSlot = 8;
        bytes32 baseSlot = keccak256(abi.encode(uint256(0), reclaimsSlot));
        // Reclaim struct layout: +0=hotkey, +1=nodeId, +2=miner, +3=amount, +4=alphaColdkey, +5=alphaAmount, +6=denyTimeout
        bytes32 amountSlot = bytes32(uint256(baseSlot) + 3);
        bytes32 alphaAmountSlot = bytes32(uint256(baseSlot) + 5);

        // Verify slot calculation by loading amount (should be 5 ether)
        assertEq(uint256(vm.load(address(collateral), amountSlot)), 5 ether);

        // Mutate reclaim 0 to be alpha-only: amount=0, alphaAmount=100 ether
        vm.store(address(collateral), amountSlot, bytes32(uint256(0)));
        vm.store(address(collateral), alphaAmountSlot, bytes32(uint256(100 ether)));

        // Also set alphaCollateralUnderPendingReclaims so deny doesn't underflow
        uint256 alphaPendingSlotIndex = 10;
        bytes32 level1 = keccak256(abi.encode(HOTKEY_1, alphaPendingSlotIndex));
        bytes32 alphaPendingSlot = keccak256(abi.encode(EXECUTOR_ID_1, level1));
        vm.store(address(collateral), alphaPendingSlot, bytes32(uint256(100 ether)));

        // Verify the reclaim now reads as alpha-only
        (,,, amt,, alphaAmt,) = collateral.reclaims(0);
        assertEq(amt, 0);
        assertEq(alphaAmt, 100 ether);

        // Deny should succeed (the fix changed `amount == 0` to `amount == 0 && alphaAmount == 0`)
        vm.prank(TRUSTEE);
        collateral.denyReclaimRequest(0, TEST_URL, TEST_SHA256);

        // Verify reclaim was cleaned up
        (,,, amt,, alphaAmt,) = collateral.reclaims(0);
        assertEq(amt, 0);
        assertEq(alphaAmt, 0);
    }

    function testFinalizeReclaimDecrementsPendingCounters() public {
        // First cycle: deposit -> reclaim -> finalize
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        vm.warp(block.timestamp + DECISION_TIMEOUT + 1);
        collateral.finalizeReclaim(0);

        // Second cycle: deposit again -> reclaim -> finalize
        // This would fail if pending counters weren't decremented in first cycle
        _seedNodeOwner(HOTKEY_1, EXECUTOR_ID_1, ALICE);
        vm.prank(ALICE);
        collateral.deposit{value: 3 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        vm.warp(block.timestamp + DECISION_TIMEOUT + 1);

        uint256 aliceBalanceBefore = ALICE.balance;
        collateral.finalizeReclaim(1);

        // Verify Alice got her 3 ETH back
        assertEq(ALICE.balance, aliceBalanceBefore + 3 ether);
        assertEq(collateral.taoCollaterals(HOTKEY_1, EXECUTOR_ID_1), 0);
    }

    function testMultiplePendingReclaimsNoTheftAfterSlash() public {
        // Regression test for P1: collateral theft via multiple pending reclaims
        // Attack path: deposit->reclaim->deposit->reclaim->slash_all->
        //   finalize clears nodeToMiner->new depositor->old finalize drains new funds

        // Step 1-2: Alice deposits 10 ETH and reclaims
        vm.prank(ALICE);
        collateral.deposit{value: 10 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        // Step 3-4: Alice deposits 5 more and reclaims the delta
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        // Step 5: Trustee slashes all 15 ETH
        vm.prank(TRUSTEE);
        collateral.slashCollateral(HOTKEY_1, EXECUTOR_ID_1, 15 ether, 0, TEST_URL, TEST_SHA256);

        // nodeToMiner must NOT be cleared (pending reclaims still exist)
        assertEq(collateral.nodeToMiner(HOTKEY_1, EXECUTOR_ID_1), ALICE);

        // Step 7: Bob tries to deposit — must fail since Alice still owns the node
        vm.prank(BOB);
        vm.expectRevert(abi.encodeWithSelector(CollateralUpgradeable.NodeNotOwned.selector));
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        // Step 8: Alice finalizes reclaim 0 — gets nothing (slashed)
        vm.warp(block.timestamp + DECISION_TIMEOUT + 1);
        collateral.finalizeReclaim(0);

        // nodeToMiner still Alice (reclaim 1 still pending)
        assertEq(collateral.nodeToMiner(HOTKEY_1, EXECUTOR_ID_1), ALICE);

        // Step 9: Alice finalizes reclaim 1 — gets nothing (slashed)
        collateral.finalizeReclaim(1);

        // NOW nodeToMiner is cleared (all pending reclaims resolved, all balances zero)
        assertEq(collateral.nodeToMiner(HOTKEY_1, EXECUTOR_ID_1), address(0));

        // Step 10: Bob can now safely deposit
        vm.prank(BOB, BOB);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 1 ether);
        assertEq(collateral.nodeToMiner(HOTKEY_1, EXECUTOR_ID_1), BOB);
        assertEq(collateral.taoCollaterals(HOTKEY_1, EXECUTOR_ID_1), 5 ether);
    }

    function testSlashAllWithPendingReclaimKeepsOwnership() public {
        // Slash all collateral while a reclaim is pending — nodeToMiner must persist
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        // Slash everything
        vm.prank(TRUSTEE);
        collateral.slashCollateral(HOTKEY_1, EXECUTOR_ID_1, 5 ether, 0, TEST_URL, TEST_SHA256);

        // nodeToMiner stays (pending reclaim exists)
        assertEq(collateral.nodeToMiner(HOTKEY_1, EXECUTOR_ID_1), ALICE);
        assertEq(collateral.taoCollaterals(HOTKEY_1, EXECUTOR_ID_1), 0);

        // Finalize reclaim — gets 0
        vm.warp(block.timestamp + DECISION_TIMEOUT + 1);
        collateral.finalizeReclaim(0);

        // NOW nodeToMiner is cleared
        assertEq(collateral.nodeToMiner(HOTKEY_1, EXECUTOR_ID_1), address(0));
    }

    // ============ DENY RECLAIM TESTS ============

    function testDenyReclaimRequest() public {
        // Setup reclaim
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        // Trustee denies
        vm.expectEmit(true, false, false, true, address(collateral));
        emit Denied(0, TEST_URL, TEST_SHA256);

        uint256 amount;
        (,,, amount,,,) = collateral.reclaims(0);
        assertEq(amount, 5 ether);

        vm.prank(TRUSTEE);
        collateral.denyReclaimRequest(0, TEST_URL, TEST_SHA256);

        // Check reclaim was deleted
        (,,, amount,,,) = collateral.reclaims(0);
        assertEq(amount, 0);
    }

    function testDenyReclaimAtExactTimeoutReverts() public {
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        uint64 denyTimeout;
        (,,,,,, denyTimeout) = collateral.reclaims(0);
        vm.warp(uint256(denyTimeout));

        vm.prank(TRUSTEE);
        vm.expectRevert(abi.encodeWithSelector(CollateralUpgradeable.PastDenyTimeout.selector));
        collateral.denyReclaimRequest(0, TEST_URL, TEST_SHA256);
    }

    function testDenyReclaimJustBeforeTimeoutSucceeds() public {
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        uint64 denyTimeout;
        (,,,,,, denyTimeout) = collateral.reclaims(0);
        vm.warp(uint256(denyTimeout) - 1);

        vm.prank(TRUSTEE);
        collateral.denyReclaimRequest(0, TEST_URL, TEST_SHA256);

        uint256 amount;
        (,,, amount,,,) = collateral.reclaims(0);
        assertEq(amount, 0);
    }

    function testDenyReclaimNotTrustee() public {
        // Setup reclaim
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        // Non-trustee tries to deny
        vm.expectRevert(
            abi.encodeWithSelector(
                bytes4(keccak256("AccessControlUnauthorizedAccount(address,bytes32)")), BOB, collateral.TRUSTEE_ROLE()
            )
        );
        vm.prank(BOB);
        collateral.denyReclaimRequest(1, TEST_URL, TEST_SHA256);
    }

    // ============ SLASH TESTS ============

    function testSlashCollateral() public {
        // Setup
        vm.prank(ALICE);
        collateral.deposit{value: 10 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        uint256 contractBalanceBefore = address(collateral).balance;

        // Slash partial amount
        vm.expectEmit(true, true, true, true, address(collateral));
        emit Slashed(HOTKEY_1, EXECUTOR_ID_1, ALICE, 5 ether, 0, TEST_URL, TEST_SHA256);

        vm.prank(TRUSTEE);
        collateral.slashCollateral(HOTKEY_1, EXECUTOR_ID_1, 5 ether, 0, TEST_URL, TEST_SHA256);

        // Check state
        assertEq(collateral.taoCollaterals(HOTKEY_1, EXECUTOR_ID_1), 5 ether);
        assertEq(address(collateral).balance, contractBalanceBefore - 5 ether);
        assertEq(TRUSTEE.balance, 5 ether);
        assertEq(collateral.nodeToMiner(HOTKEY_1, EXECUTOR_ID_1), ALICE); // Still owned
    }

    function testSlashAllCollateral() public {
        // Setup
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        // Slash all
        vm.prank(TRUSTEE);
        collateral.slashCollateral(HOTKEY_1, EXECUTOR_ID_1, 5 ether, 0, TEST_URL, TEST_SHA256);

        // Check executor ownership is cleared
        assertEq(collateral.nodeToMiner(HOTKEY_1, EXECUTOR_ID_1), address(0));
        assertEq(collateral.taoCollaterals(HOTKEY_1, EXECUTOR_ID_1), 0);
    }

    function testSlashNotTrustee() public {
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.expectRevert(
            abi.encodeWithSelector(
                bytes4(keccak256("AccessControlUnauthorizedAccount(address,bytes32)")), BOB, collateral.TRUSTEE_ROLE()
            )
        );
        vm.prank(BOB);
        collateral.slashCollateral(HOTKEY_1, EXECUTOR_ID_1, 5 ether, 0, TEST_URL, TEST_SHA256);
    }

    function testSlashAmountZero() public {
        vm.prank(TRUSTEE);
        vm.expectRevert(abi.encodeWithSelector(CollateralUpgradeable.AmountZero.selector));
        collateral.slashCollateral(HOTKEY_1, EXECUTOR_ID_1, 0, 0, TEST_URL, TEST_SHA256);
    }

    // ============ ADMIN FUNCTION TESTS ============

    function testUpdateTrustee() public {
        address newTrustee = makeAddr("newTrustee");

        vm.expectEmit(true, true, false, false, address(collateral));
        emit TrusteeUpdated(TRUSTEE, newTrustee);

        vm.prank(ADMIN);
        collateral.updateTrustee(newTrustee);

        assertEq(collateral.trustee(), newTrustee);
        assertTrue(collateral.hasRole(collateral.TRUSTEE_ROLE(), newTrustee));
        assertFalse(collateral.hasRole(collateral.TRUSTEE_ROLE(), TRUSTEE));
    }

    // --- TRUSTEE_ROLE direct modification prevention tests ---

    function testGrantRoleTrusteeRoleReverts() public {
        address attacker = makeAddr("attacker");
        bytes32 trusteeRole = collateral.TRUSTEE_ROLE();
        vm.prank(ADMIN);
        vm.expectRevert(CollateralUpgradeable.TrusteeRoleDirectModificationForbidden.selector);
        collateral.grantRole(trusteeRole, attacker);
    }

    function testRevokeRoleTrusteeRoleReverts() public {
        bytes32 trusteeRole = collateral.TRUSTEE_ROLE();
        vm.prank(ADMIN);
        vm.expectRevert(CollateralUpgradeable.TrusteeRoleDirectModificationForbidden.selector);
        collateral.revokeRole(trusteeRole, TRUSTEE);
    }

    function testRenounceRoleTrusteeRoleReverts() public {
        bytes32 trusteeRole = collateral.TRUSTEE_ROLE();
        vm.prank(TRUSTEE);
        vm.expectRevert(CollateralUpgradeable.TrusteeRoleDirectModificationForbidden.selector);
        collateral.renounceRole(trusteeRole, TRUSTEE);
    }

    function testGrantRoleNonTrusteeStillWorks() public {
        address newUpgrader = makeAddr("newUpgrader");
        bytes32 upgraderRole = collateral.UPGRADER_ROLE();
        vm.prank(ADMIN);
        collateral.grantRole(upgraderRole, newUpgrader);
        assertTrue(collateral.hasRole(upgraderRole, newUpgrader));
    }

    function testRevokeRoleNonTrusteeStillWorks() public {
        bytes32 upgraderRole = collateral.UPGRADER_ROLE();
        // ADMIN already has UPGRADER_ROLE from initialize
        vm.prank(ADMIN);
        collateral.revokeRole(upgraderRole, ADMIN);
        assertFalse(collateral.hasRole(upgraderRole, ADMIN));
    }

    function testRenounceRoleNonTrusteeStillWorks() public {
        bytes32 adminRole = collateral.DEFAULT_ADMIN_ROLE();
        // ADMIN has DEFAULT_ADMIN_ROLE from initialize
        vm.prank(ADMIN);
        collateral.renounceRole(adminRole, ADMIN);
        assertFalse(collateral.hasRole(adminRole, ADMIN));
    }

    function testUpdateTrusteeStillWorksAfterOverrides() public {
        address newTrustee = makeAddr("newTrustee2");
        bytes32 trusteeRole = collateral.TRUSTEE_ROLE();
        vm.prank(ADMIN);
        collateral.updateTrustee(newTrustee);

        assertEq(collateral.trustee(), newTrustee);
        assertTrue(collateral.hasRole(trusteeRole, newTrustee));
        assertFalse(collateral.hasRole(trusteeRole, TRUSTEE));
    }

    function testUpdateDecisionTimeout() public {
        vm.prank(ADMIN);
        collateral.updateDecisionTimeout(7200);

        assertEq(collateral.decisionTimeout(), 7200);
    }

    function testUpdateMinCollateralIncrease() public {
        vm.prank(ADMIN);
        collateral.updateMinCollateralIncrease(2 ether);

        assertEq(collateral.minCollateralIncrease(), 2 ether);
    }

    function testSetContractColdkeyFunctionRemoved() public {
        (bool success, bytes memory returndata) =
            address(collateral).call(abi.encodeWithSignature("setContractColdkey(bytes32)", bytes32(uint256(999))));

        assertFalse(success);
        assertGe(returndata.length, 4);
        bytes4 selector;
        assembly {
            selector := mload(add(returndata, 0x20))
        }
        assertEq(selector, CollateralUpgradeable.InvalidDepositMethod.selector);
        assertEq(collateral.contractColdkey(), bytes32(uint256(uint160(address(proxy)))));
    }

    // ============ DEPOSIT TYPE TOGGLE TESTS ============

    function testDepositRevertsTaoDepositsDisabled() public {
        vm.prank(ADMIN);
        collateral.updateTaoDepositsEnabled(false);

        vm.prank(ALICE);
        vm.expectRevert(abi.encodeWithSelector(CollateralUpgradeable.TaoDepositsDisabled.selector));
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);
    }

    function testDepositRevertsAlphaDepositsDisabled() public {
        vm.prank(ADMIN);
        collateral.updateAlphaDepositsEnabled(false);

        vm.prank(ALICE);
        vm.expectRevert(abi.encodeWithSelector(CollateralUpgradeable.AlphaDepositsDisabled.selector));
        collateral.deposit(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 1);
    }

    function testAlphaOnlyFirstDepositSucceedsWhenOnlyAlphaEnabled() public {
        vm.prank(ADMIN);
        collateral.updateTaoDepositsEnabled(false);

        vm.prank(ALICE, ALICE);
        collateral.deposit(HOTKEY_2, EXECUTOR_ID_2, ALPHA_HOTKEY, 1 ether);
        assertEq(collateral.nodeToMiner(HOTKEY_2, EXECUTOR_ID_2), ALICE);
        assertGt(collateral.alphaCollaterals(HOTKEY_2, EXECUTOR_ID_2), 0);
    }

    function testReclaimStillWorksAfterDisablingDepositedType() public {
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        // Disable TAO deposits
        vm.prank(ADMIN);
        collateral.updateTaoDepositsEnabled(false);

        // Reclaim should still work
        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        vm.warp(block.timestamp + DECISION_TIMEOUT + 1);
        uint256 aliceBalanceBefore = ALICE.balance;
        collateral.finalizeReclaim(0);
        assertEq(ALICE.balance, aliceBalanceBefore + 5 ether);
    }

    function testSlashStillWorksAfterDisablingDepositedType() public {
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        // Disable TAO deposits
        vm.prank(ADMIN);
        collateral.updateTaoDepositsEnabled(false);

        // Slash should still work
        vm.prank(TRUSTEE);
        collateral.slashCollateral(HOTKEY_1, EXECUTOR_ID_1, 5 ether, 0, TEST_URL, TEST_SHA256);
        assertEq(collateral.taoCollaterals(HOTKEY_1, EXECUTOR_ID_1), 0);
    }

    function testAlphaReclaimStillWorksAfterDisablingAlphaDeposits() public {
        vm.prank(ALICE, ALICE);
        collateral.deposit(HOTKEY_2, EXECUTOR_ID_2, ALPHA_HOTKEY, 1 ether);
        uint256 depositedAlpha = collateral.alphaCollaterals(HOTKEY_2, EXECUTOR_ID_2);
        assertGt(depositedAlpha, 0);

        vm.prank(ADMIN);
        collateral.updateAlphaDepositsEnabled(false);

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_2, EXECUTOR_ID_2);
        (,,, uint256 taoReclaimAmount,, uint256 alphaReclaimAmount,) = collateral.reclaims(0);
        assertEq(taoReclaimAmount, 0);
        assertEq(alphaReclaimAmount, depositedAlpha);

        vm.warp(block.timestamp + DECISION_TIMEOUT + 1);
        collateral.finalizeReclaim(0);
        assertEq(collateral.alphaCollaterals(HOTKEY_2, EXECUTOR_ID_2), 0);
        assertEq(collateral.nodeToMiner(HOTKEY_2, EXECUTOR_ID_2), address(0));
    }

    function testAlphaSlashStillWorksAfterDisablingAlphaDeposits() public {
        vm.prank(ALICE, ALICE);
        collateral.deposit(HOTKEY_2, EXECUTOR_ID_2, ALPHA_HOTKEY, 1 ether);
        uint256 depositedAlpha = collateral.alphaCollaterals(HOTKEY_2, EXECUTOR_ID_2);
        assertGt(depositedAlpha, 0);

        vm.prank(ADMIN);
        collateral.updateAlphaDepositsEnabled(false);

        vm.prank(TRUSTEE);
        collateral.slashCollateral(HOTKEY_2, EXECUTOR_ID_2, 0, depositedAlpha, TEST_URL, TEST_SHA256);
        assertEq(collateral.alphaCollaterals(HOTKEY_2, EXECUTOR_ID_2), 0);
        assertEq(collateral.nodeToMiner(HOTKEY_2, EXECUTOR_ID_2), address(0));
    }

    function testOnlyAdminCanCallToggleFunctions() public {
        vm.expectRevert(
            abi.encodeWithSelector(
                bytes4(keccak256("AccessControlUnauthorizedAccount(address,bytes32)")),
                ALICE,
                collateral.DEFAULT_ADMIN_ROLE()
            )
        );
        vm.prank(ALICE);
        collateral.updateTaoDepositsEnabled(false);

        vm.expectRevert(
            abi.encodeWithSelector(
                bytes4(keccak256("AccessControlUnauthorizedAccount(address,bytes32)")),
                ALICE,
                collateral.DEFAULT_ADMIN_ROLE()
            )
        );
        vm.prank(ALICE);
        collateral.updateAlphaDepositsEnabled(false);
    }

    function testToggleEventsEmittedCorrectly() public {
        vm.expectEmit(false, false, false, true, address(collateral));
        emit TaoDepositsEnabledUpdated(false);
        vm.prank(ADMIN);
        collateral.updateTaoDepositsEnabled(false);

        vm.expectEmit(false, false, false, true, address(collateral));
        emit AlphaDepositsEnabledUpdated(true);
        vm.prank(ADMIN);
        collateral.updateAlphaDepositsEnabled(true);
    }

    function testInitializationSetsDepositToggles() public view {
        assertTrue(collateral.taoDepositsEnabled());
        assertTrue(collateral.alphaDepositsEnabled());
    }

    function testInitializationSetsDepositTogglesFalseFalse() public {
        CollateralUpgradeable collateralWithDisabledDeposits = _deployCollateralWithDepositToggles(false, false);
        assertFalse(collateralWithDisabledDeposits.taoDepositsEnabled());
        assertFalse(collateralWithDisabledDeposits.alphaDepositsEnabled());
    }

    function testInitializationSetsDepositTogglesMixed() public {
        CollateralUpgradeable taoOnlyCollateral = _deployCollateralWithDepositToggles(true, false);
        assertTrue(taoOnlyCollateral.taoDepositsEnabled());
        assertFalse(taoOnlyCollateral.alphaDepositsEnabled());

        CollateralUpgradeable alphaOnlyCollateral = _deployCollateralWithDepositToggles(false, true);
        assertFalse(alphaOnlyCollateral.taoDepositsEnabled());
        assertTrue(alphaOnlyCollateral.alphaDepositsEnabled());
    }

    // ============ ACTIVE NODE TRACKING TESTS ============

    function testGetAllCollateralsEmptyInitially() public view {
        CollateralUpgradeable.NodeCollateral[] memory results = _allCollaterals();
        assertEq(results.length, 0);
        assertEq(collateral.getActiveNodeCount(), 0);
    }

    function testGetAllCollateralsAfterDeposit() public {
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        CollateralUpgradeable.NodeCollateral[] memory results = _allCollaterals();
        assertEq(results.length, 1);
        assertEq(results[0].hotkey, HOTKEY_1);
        assertEq(results[0].nodeId, EXECUTOR_ID_1);
        assertEq(results[0].miner, ALICE);
        assertEq(results[0].taoCollateral, 5 ether);
        assertEq(results[0].alphaCollateral, 0);
        assertEq(collateral.getActiveNodeCount(), 1);
    }

    function testGetAllCollateralsMultipleNodes() public {
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(BOB, BOB);
        collateral.deposit{value: 3 ether}(HOTKEY_2, EXECUTOR_ID_2, ALPHA_HOTKEY, 0);

        CollateralUpgradeable.NodeCollateral[] memory results = _allCollaterals();
        assertEq(results.length, 2);
        assertEq(collateral.getActiveNodeCount(), 2);
    }

    function testNodeRemovedFromAllCollateralsAfterFullSlash() public {
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);
        assertEq(collateral.getActiveNodeCount(), 1);

        vm.prank(TRUSTEE);
        collateral.slashCollateral(HOTKEY_1, EXECUTOR_ID_1, 5 ether, 0, TEST_URL, TEST_SHA256);

        assertEq(collateral.getActiveNodeCount(), 0);
        CollateralUpgradeable.NodeCollateral[] memory results = _allCollaterals();
        assertEq(results.length, 0);
    }

    function testNodeNotRemovedAfterPartialSlash() public {
        vm.prank(ALICE);
        collateral.deposit{value: 10 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(TRUSTEE);
        collateral.slashCollateral(HOTKEY_1, EXECUTOR_ID_1, 5 ether, 0, TEST_URL, TEST_SHA256);

        assertEq(collateral.getActiveNodeCount(), 1);
        CollateralUpgradeable.NodeCollateral[] memory results = _allCollaterals();
        assertEq(results.length, 1);
        assertEq(results[0].taoCollateral, 5 ether);
    }

    function testNodeRemovedAfterFullReclaim() public {
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        vm.warp(block.timestamp + DECISION_TIMEOUT + 1);
        collateral.finalizeReclaim(0);

        assertEq(collateral.getActiveNodeCount(), 0);
    }

    function testNodeRemovedAfterDenyWithZeroBalance() public {
        // Deposit, reclaim, slash all, then deny — should remove node
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        vm.prank(TRUSTEE);
        collateral.slashCollateral(HOTKEY_1, EXECUTOR_ID_1, 5 ether, 0, TEST_URL, TEST_SHA256);

        // Node still active (pending reclaim)
        assertEq(collateral.getActiveNodeCount(), 1);

        vm.prank(TRUSTEE);
        collateral.denyReclaimRequest(0, TEST_URL, TEST_SHA256);

        // Now everything is zero
        assertEq(collateral.getActiveNodeCount(), 0);
    }

    function testSwapAndPopOrdering() public {
        // Deposit A, B, C
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(BOB, BOB);
        collateral.deposit{value: 3 ether}(HOTKEY_2, EXECUTOR_ID_2, ALPHA_HOTKEY, 0);

        bytes32 HOTKEY_3 = bytes32(uint256(102));
        bytes16 EXECUTOR_ID_3 = bytes16(uint128(3));
        _seedNodeOwner(HOTKEY_3, EXECUTOR_ID_3, CHARLIE);
        vm.prank(CHARLIE);
        collateral.deposit{value: 2 ether}(HOTKEY_3, EXECUTOR_ID_3, ALPHA_HOTKEY, 0);

        assertEq(collateral.getActiveNodeCount(), 3);

        // Slash B (middle element)
        vm.prank(TRUSTEE);
        collateral.slashCollateral(HOTKEY_2, EXECUTOR_ID_2, 3 ether, 0, TEST_URL, TEST_SHA256);

        // Should have A and C remaining
        assertEq(collateral.getActiveNodeCount(), 2);
        CollateralUpgradeable.NodeCollateral[] memory results = _allCollaterals();
        assertEq(results.length, 2);

        // After swap-and-pop: A should be at index 0, C should be at index 1
        assertEq(results[0].hotkey, HOTKEY_1);
        assertEq(results[1].hotkey, HOTKEY_3);
    }

    function testDuplicateDepositDoesNotDuplicateTracking() public {
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(ALICE);
        collateral.deposit{value: 3 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        assertEq(collateral.getActiveNodeCount(), 1);
        CollateralUpgradeable.NodeCollateral[] memory results = _allCollaterals();
        assertEq(results.length, 1);
        assertEq(results[0].taoCollateral, 8 ether);
    }

    // ============ ACTIVE RECLAIM TRACKING TESTS ============

    function testGetAllReclaimsEmptyInitially() public view {
        CollateralUpgradeable.ReclaimInfo[] memory results = _allReclaims();
        assertEq(results.length, 0);
        assertEq(collateral.getActiveReclaimCount(), 0);
    }

    function testGetAllReclaimsAfterReclaimStarted() public {
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        CollateralUpgradeable.ReclaimInfo[] memory results = _allReclaims();
        assertEq(results.length, 1);
        assertEq(results[0].reclaimRequestId, 0);
        assertEq(results[0].hotkey, HOTKEY_1);
        assertEq(results[0].nodeId, EXECUTOR_ID_1);
        assertEq(results[0].miner, ALICE);
        assertEq(results[0].amount, 5 ether);
        assertEq(collateral.getActiveReclaimCount(), 1);
    }

    function testReclaimRemovedAfterFinalize() public {
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        vm.warp(block.timestamp + DECISION_TIMEOUT + 1);
        collateral.finalizeReclaim(0);

        assertEq(collateral.getActiveReclaimCount(), 0);
        CollateralUpgradeable.ReclaimInfo[] memory results = _allReclaims();
        assertEq(results.length, 0);
    }

    function testReclaimRemovedAfterDeny() public {
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        vm.prank(TRUSTEE);
        collateral.denyReclaimRequest(0, TEST_URL, TEST_SHA256);

        assertEq(collateral.getActiveReclaimCount(), 0);
    }

    // ============ PAGINATION TESTS ============

    function testGetAllCollateralsPaginated() public {
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(BOB, BOB);
        collateral.deposit{value: 3 ether}(HOTKEY_2, EXECUTOR_ID_2, ALPHA_HOTKEY, 0);

        // Page 1: offset=0, limit=1
        CollateralUpgradeable.NodeCollateral[] memory page1 = collateral.getAllCollaterals(0, 1);
        assertEq(page1.length, 1);
        assertEq(page1[0].hotkey, HOTKEY_1);

        // Page 2: offset=1, limit=1
        CollateralUpgradeable.NodeCollateral[] memory page2 = collateral.getAllCollaterals(1, 1);
        assertEq(page2.length, 1);
        assertEq(page2[0].hotkey, HOTKEY_2);

        // Offset beyond array
        CollateralUpgradeable.NodeCollateral[] memory empty = collateral.getAllCollaterals(10, 5);
        assertEq(empty.length, 0);

        // Zero limit
        CollateralUpgradeable.NodeCollateral[] memory zeroLimit = collateral.getAllCollaterals(0, 0);
        assertEq(zeroLimit.length, 0);

        // Limit larger than remaining
        CollateralUpgradeable.NodeCollateral[] memory all = collateral.getAllCollaterals(0, 100);
        assertEq(all.length, 2);

        // Max limit should not overflow and should clamp to remaining
        CollateralUpgradeable.NodeCollateral[] memory maxLimit =
            collateral.getAllCollaterals(1, type(uint256).max);
        assertEq(maxLimit.length, 1);
        assertEq(maxLimit[0].hotkey, HOTKEY_2);
    }

    function testGetAllReclaimsPaginated() public {
        vm.prank(ALICE);
        collateral.deposit{value: 10 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        CollateralUpgradeable.ReclaimInfo[] memory page = collateral.getAllReclaims(0, 1);
        assertEq(page.length, 1);
        assertEq(page[0].reclaimRequestId, 0);

        CollateralUpgradeable.ReclaimInfo[] memory page2 = collateral.getAllReclaims(1, 1);
        assertEq(page2.length, 1);
        assertEq(page2[0].reclaimRequestId, 1);

        CollateralUpgradeable.ReclaimInfo[] memory empty = collateral.getAllReclaims(10, 5);
        assertEq(empty.length, 0);

        CollateralUpgradeable.ReclaimInfo[] memory zeroLimit = collateral.getAllReclaims(0, 0);
        assertEq(zeroLimit.length, 0);

        CollateralUpgradeable.ReclaimInfo[] memory maxLimit = collateral.getAllReclaims(1, type(uint256).max);
        assertEq(maxLimit.length, 1);
        assertEq(maxLimit[0].reclaimRequestId, 1);
    }

    // ============ UPGRADE TESTS ============

    function testUpgrade() public {
        CollateralUpgradeableUpgradeMock newImplementation = new CollateralUpgradeableUpgradeMock();

        vm.expectEmit(true, true, false, false, address(collateral));
        emit ContractUpgraded(2, address(newImplementation));

        vm.prank(ADMIN);
        collateral.upgradeToAndCall(address(newImplementation), "");

        assertEq(collateral.getVersion(), 2);
    }

    // ============ EVENTS ============

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
        uint64 expirationTime
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

    event Denied(uint256 indexed reclaimRequestId, string url, bytes32 urlContentSha256);

    event Slashed(
        bytes32 indexed hotkey,
        bytes16 indexed executorId,
        address indexed miner,
        uint256 slashAmount,
        uint256 slashAlphaAmount,
        string url,
        bytes32 urlContentSha256
    );

    event ContractUpgraded(uint256 indexed newVersion, address indexed newImplementation);

    event TrusteeUpdated(address indexed oldTrustee, address indexed newTrustee);

    event TaoDepositsEnabledUpdated(bool enabled);
    event AlphaDepositsEnabledUpdated(bool enabled);
}
