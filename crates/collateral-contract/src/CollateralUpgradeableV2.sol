// SPDX-License-Identifier: UNLICENSED

pragma solidity ^0.8.22;

import "./CollateralUpgradeable.sol";

/// @notice V2 test implementation used to exercise proxy upgrades while keeping
/// storage layout fully compatible with CollateralUpgradeable.
contract CollateralUpgradeableV2 is CollateralUpgradeable {
    function getVersion() external pure override returns (uint256) {
        return 2;
    }

    function initializeV2() external reinitializer(2) {}
}
