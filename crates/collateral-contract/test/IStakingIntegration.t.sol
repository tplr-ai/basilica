// SPDX-License-Identifier: UNLICENSED

pragma solidity ^0.8.24;

import {Test} from "forge-std/Test.sol";
import {CollateralUpgradeable} from "../src/CollateralUpgradeable.sol";
import {ERC1967Proxy} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";

contract AddressMappingPrecompileMock {
    function addressMapping(address evmAddress) external pure returns (bytes32) {
        return bytes32(uint256(uint160(evmAddress)));
    }
}

// Mock IStaking precompile that works with both call and delegatecall entrypoints.
contract AdvancedMockIStaking {
    address private constant ISTAKING_PRECOMPILE = 0x0000000000000000000000000000000000000805;

    // coldkey -> hotkey -> netuid -> amount
    mapping(bytes32 => mapping(bytes32 => mapping(uint256 => uint256))) public stakes;

    function transferStake(bytes32 toColdkey, bytes32 hotkey, uint256 fromNetuid, uint256 toNetuid, uint256 amount)
        external
    {
        bytes32 fromColdkey = bytes32(uint256(uint160(msg.sender)));

        if (address(this) == ISTAKING_PRECOMPILE) {
            _applyTransferStake(fromColdkey, toColdkey, hotkey, fromNetuid, toNetuid, amount);
            return;
        }

        (bool ok,) = ISTAKING_PRECOMPILE.call(
            abi.encodeWithSelector(
                this.applyTransferStake.selector, fromColdkey, toColdkey, hotkey, fromNetuid, toNetuid, amount
            )
        );
        require(ok, "forward transfer failed");
    }

    function applyTransferStake(
        bytes32 fromColdkey,
        bytes32 toColdkey,
        bytes32 hotkey,
        uint256 fromNetuid,
        uint256 toNetuid,
        uint256 amount
    ) external {
        require(address(this) == ISTAKING_PRECOMPILE, "must run at precompile");
        _applyTransferStake(fromColdkey, toColdkey, hotkey, fromNetuid, toNetuid, amount);
    }

    function moveStake(bytes32 fromHotkey, bytes32 toHotkey, uint256 fromNetuid, uint256 toNetuid, uint256 amount)
        external
    {
        bytes32 coldkey = bytes32(uint256(uint160(msg.sender)));

        if (address(this) == ISTAKING_PRECOMPILE) {
            _applyMoveStake(coldkey, fromHotkey, toHotkey, fromNetuid, toNetuid, amount);
            return;
        }

        (bool ok,) = ISTAKING_PRECOMPILE.call(
            abi.encodeWithSelector(
                this.applyMoveStake.selector, coldkey, fromHotkey, toHotkey, fromNetuid, toNetuid, amount
            )
        );
        require(ok, "forward move failed");
    }

    function applyMoveStake(
        bytes32 coldkey,
        bytes32 fromHotkey,
        bytes32 toHotkey,
        uint256 fromNetuid,
        uint256 toNetuid,
        uint256 amount
    ) external {
        require(address(this) == ISTAKING_PRECOMPILE, "must run at precompile");
        _applyMoveStake(coldkey, fromHotkey, toHotkey, fromNetuid, toNetuid, amount);
    }

    function getStake(bytes32 hotkey, bytes32 coldkey, uint256 netuid) external view returns (uint256) {
        return stakes[coldkey][hotkey][netuid];
    }

    // Helper functions for tests.
    function setStake(bytes32 hotkey, bytes32 coldkey, uint256 netuid, uint256 amount) external {
        stakes[coldkey][hotkey][netuid] = amount;
    }

    function _applyTransferStake(
        bytes32 fromColdkey,
        bytes32 toColdkey,
        bytes32 hotkey,
        uint256 fromNetuid,
        uint256 toNetuid,
        uint256 amount
    ) internal {
        require(stakes[fromColdkey][hotkey][fromNetuid] >= amount, "insufficient stake");
        stakes[fromColdkey][hotkey][fromNetuid] -= amount;
        stakes[toColdkey][hotkey][toNetuid] += amount;
    }

    function _applyMoveStake(
        bytes32 coldkey,
        bytes32 fromHotkey,
        bytes32 toHotkey,
        uint256 fromNetuid,
        uint256 toNetuid,
        uint256 amount
    ) internal {
        require(stakes[coldkey][fromHotkey][fromNetuid] >= amount, "insufficient stake");
        stakes[coldkey][fromHotkey][fromNetuid] -= amount;
        stakes[coldkey][toHotkey][toNetuid] += amount;
    }
}

contract IStakingIntegrationTest is Test {
    address constant ISTAKING_PRECOMPILE = 0x0000000000000000000000000000000000000805;
    address constant ADDRESS_MAPPING_PRECOMPILE = 0x000000000000000000000000000000000000080C;

    CollateralUpgradeable public collateral;
    AdvancedMockIStaking public mockStaking;

    // Test parameters
    uint16 constant NETUID = 39;

    uint256 constant MIN_DEPOSIT = 1 ether;
    uint64 constant DECISION_TIMEOUT = 3600;

    address constant ALICE = address(0x09);
    bytes32 constant ALICE_COLDKEY = bytes32(uint256(9));

    bytes32 contractColdkey;
    bytes32 constant VALIDATOR_HOTKEY = bytes32(uint256(88));

    bytes32 constant HOTKEY_1 = bytes32(uint256(101));
    bytes16 constant EXECUTOR_ID_1 = bytes16(uint128(1));

    address constant TRUSTEE = address(0x1111);
    address constant ADMIN = address(0x2222);

    uint256 constant ALPHA_AMOUNT = 5 ether;
    string constant TEST_URL = "https://example.com/reclaim";
    bytes32 constant TEST_SHA256 = bytes32(0x1234567890123456789012345678901212345678901234567890123456789012);

    function setUp() public {
        // Deploy precompile mocks.
        AddressMappingPrecompileMock addressMappingMock = new AddressMappingPrecompileMock();
        AdvancedMockIStaking mockStakingCode = new AdvancedMockIStaking();

        vm.etch(ADDRESS_MAPPING_PRECOMPILE, address(addressMappingMock).code);
        vm.etch(ISTAKING_PRECOMPILE, address(mockStakingCode).code);
        mockStaking = AdvancedMockIStaking(ISTAKING_PRECOMPILE);

        // Deploy collateral contract behind proxy.
        CollateralUpgradeable implementation = new CollateralUpgradeable();
        bytes memory initData = abi.encodeWithSelector(
            CollateralUpgradeable.initialize.selector,
            NETUID,
            TRUSTEE,
            MIN_DEPOSIT,
            DECISION_TIMEOUT,
            ADMIN,
            VALIDATOR_HOTKEY,
            true,
            true
        );
        ERC1967Proxy proxy = new ERC1967Proxy(address(implementation), initData);
        collateral = CollateralUpgradeable(payable(address(proxy)));

        contractColdkey = bytes32(uint256(uint160(address(proxy))));

        vm.deal(ALICE, 100 ether);
    }

    function testAlphaDepositReclaimFlowWithoutManualColdkeyConfiguration() public {
        assertEq(collateral.contractColdkey(), contractColdkey);

        // Seed Alice alpha on the same hotkey to avoid moveStake path noise.
        mockStaking.setStake(VALIDATOR_HOTKEY, ALICE_COLDKEY, NETUID, ALPHA_AMOUNT);

        vm.prank(ALICE, ALICE);
        collateral.deposit(HOTKEY_1, EXECUTOR_ID_1, VALIDATOR_HOTKEY, ALPHA_AMOUNT);

        assertEq(collateral.alphaCollaterals(HOTKEY_1, EXECUTOR_ID_1), ALPHA_AMOUNT);
        assertEq(mockStaking.getStake(VALIDATOR_HOTKEY, ALICE_COLDKEY, NETUID), 0);
        assertEq(mockStaking.getStake(VALIDATOR_HOTKEY, contractColdkey, NETUID), ALPHA_AMOUNT);

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1, TEST_URL, TEST_SHA256);

        vm.warp(block.timestamp + DECISION_TIMEOUT + 1);
        collateral.finalizeReclaim(0);

        assertEq(mockStaking.getStake(VALIDATOR_HOTKEY, contractColdkey, NETUID), 0);
        assertEq(mockStaking.getStake(VALIDATOR_HOTKEY, ALICE_COLDKEY, NETUID), ALPHA_AMOUNT);
    }

    function testSlashAlphaTransfersStakeToTrusteeColdkey() public {
        bytes32 trusteeColdkey = bytes32(uint256(uint160(TRUSTEE)));
        uint256 slashAlphaAmount = 2 ether;

        mockStaking.setStake(VALIDATOR_HOTKEY, ALICE_COLDKEY, NETUID, ALPHA_AMOUNT);

        vm.prank(ALICE, ALICE);
        collateral.deposit(HOTKEY_1, EXECUTOR_ID_1, VALIDATOR_HOTKEY, ALPHA_AMOUNT);

        assertEq(mockStaking.getStake(VALIDATOR_HOTKEY, contractColdkey, NETUID), ALPHA_AMOUNT);
        assertEq(mockStaking.getStake(VALIDATOR_HOTKEY, trusteeColdkey, NETUID), 0);

        vm.prank(TRUSTEE, TRUSTEE);
        collateral.slashCollateral(HOTKEY_1, EXECUTOR_ID_1, 0, slashAlphaAmount, TEST_URL, TEST_SHA256);

        assertEq(collateral.alphaCollaterals(HOTKEY_1, EXECUTOR_ID_1), ALPHA_AMOUNT - slashAlphaAmount);
        assertEq(mockStaking.getStake(VALIDATOR_HOTKEY, contractColdkey, NETUID), ALPHA_AMOUNT - slashAlphaAmount);
        assertEq(mockStaking.getStake(VALIDATOR_HOTKEY, trusteeColdkey, NETUID), slashAlphaAmount);
    }
}
