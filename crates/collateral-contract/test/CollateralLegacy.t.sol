// SPDX-License-Identifier: UNLICENSED

pragma solidity ^0.8.22;

import {Test} from "forge-std/Test.sol";
import {Collateral} from "../src/Collateral.sol";

contract ContractDepositorLegacy {
    function claimNode(
        Collateral collateral,
        bytes32 hotkey,
        bytes16 nodeId
    ) external payable {
        collateral.deposit{value: msg.value}(hotkey, nodeId);
    }
}

contract CollateralLegacyTest is Test {
    Collateral public collateral;

    uint16 constant NETUID = 42;
    address constant TRUSTEE = address(0x123);
    uint256 constant MIN_DEPOSIT = 1 ether;
    uint64 constant DECISION_TIMEOUT = 3600;

    address constant ALICE = address(0x789);
    bytes32 constant HOTKEY = bytes32(uint256(100));
    bytes16 constant NODE_ID = bytes16(uint128(1));

    string constant TEST_URL = "https://example.com/proof";
    bytes32 constant TEST_SHA256 =
        bytes32(0x1234567890123456789012345678901212345678901234567890123456789012);

    function setUp() public {
        collateral = new Collateral(NETUID, TRUSTEE, MIN_DEPOSIT, DECISION_TIMEOUT);
        vm.deal(ALICE, 100 ether);
    }

    function _depositAndReclaim() internal {
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY, NODE_ID);

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY, NODE_ID, TEST_URL, TEST_SHA256);
    }

    function testFirstOwnershipClaimMustBeEOA() public {
        ContractDepositorLegacy depositor = new ContractDepositorLegacy();

        vm.expectRevert(abi.encodeWithSelector(Collateral.MinerMustBeEOA.selector));
        depositor.claimNode{value: 1 ether}(collateral, HOTKEY, NODE_ID);
    }

    function testFinalizeReclaimPartialAfterSlash() public {
        _depositAndReclaim();

        vm.prank(TRUSTEE);
        collateral.slashCollateralAmount(HOTKEY, NODE_ID, 3 ether, TEST_URL, TEST_SHA256);

        vm.warp(block.timestamp + DECISION_TIMEOUT + 1);
        uint256 aliceBalanceBefore = ALICE.balance;

        collateral.finalizeReclaim(1);

        assertEq(ALICE.balance, aliceBalanceBefore + 2 ether);
        assertEq(collateral.collaterals(HOTKEY, NODE_ID), 0);
        assertEq(collateral.nodeToMiner(HOTKEY, NODE_ID), address(0));
    }

    function testFinalizeReclaimAfterFullSlashCleansUpRequest() public {
        _depositAndReclaim();

        vm.prank(TRUSTEE);
        collateral.slashCollateral(HOTKEY, NODE_ID, TEST_URL, TEST_SHA256);

        // Pending reclaim still exists, so ownership must remain.
        assertEq(collateral.nodeToMiner(HOTKEY, NODE_ID), ALICE);

        vm.warp(block.timestamp + DECISION_TIMEOUT + 1);
        uint256 aliceBalanceBefore = ALICE.balance;

        collateral.finalizeReclaim(1);

        assertEq(ALICE.balance, aliceBalanceBefore);
        assertEq(collateral.collaterals(HOTKEY, NODE_ID), 0);
        assertEq(collateral.nodeToMiner(HOTKEY, NODE_ID), address(0));
        (,,,uint256 amount,) = collateral.reclaims(1);
        assertEq(amount, 0);
    }

    function testReclaimAfterFullSlashReturnsAmountZeroNotUnderflow() public {
        _depositAndReclaim();

        vm.prank(TRUSTEE);
        collateral.slashCollateral(HOTKEY, NODE_ID, TEST_URL, TEST_SHA256);

        vm.prank(ALICE);
        vm.expectRevert(abi.encodeWithSelector(Collateral.AmountZero.selector));
        collateral.reclaimCollateral(HOTKEY, NODE_ID, TEST_URL, TEST_SHA256);
    }

    function testDenyAfterFullSlashClearsOwnership() public {
        _depositAndReclaim();

        vm.prank(TRUSTEE);
        collateral.slashCollateral(HOTKEY, NODE_ID, TEST_URL, TEST_SHA256);
        assertEq(collateral.nodeToMiner(HOTKEY, NODE_ID), ALICE);

        vm.prank(TRUSTEE);
        collateral.denyReclaimRequest(1, TEST_URL, TEST_SHA256);

        assertEq(collateral.nodeToMiner(HOTKEY, NODE_ID), address(0));
    }
}
