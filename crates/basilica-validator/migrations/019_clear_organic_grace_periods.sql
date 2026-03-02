-- Clear stale organic grace period entries so only future force_exclude (slash)
-- entries remain. Without this, old organic entries with expired grace periods
-- would incorrectly appear as slash-excluded nodes.
DELETE FROM collateral_grace_periods;
