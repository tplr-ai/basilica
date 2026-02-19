// SPDX-License-Identifier: UNLICENSED

pragma solidity ^0.8.22;

import {CollateralUpgradeable} from "../../src/CollateralUpgradeable.sol";

/// @notice Test-only upgrade mock used to exercise proxy upgrades while keeping
/// storage layout fully compatible with CollateralUpgradeable.
contract CollateralUpgradeableUpgradeMock is CollateralUpgradeable {
    function getVersion() external pure override returns (uint256) {
        return 2;
    }

    function initializeUpgradeMock() external reinitializer(2) {}
}
