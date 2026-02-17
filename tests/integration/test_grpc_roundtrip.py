"""Integration tests for gRPC roundtrip between Draft and Target workers.

These tests require a running gRPC server or will spin one up in-process.
They are marked with ``@pytest.mark.integration`` and excluded from the
default test run (use ``make test-all`` to include them).

.. todo::
    Implement full gRPC integration tests once protobuf stubs are generated.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
class TestGRPCRoundtrip:
    """End-to-end gRPC tests for the draft → target pipeline."""

    def test_draft_service_ping(self):
        """Draft service should respond to health check.

        .. todo::
            Start an in-process Draft gRPC server and call Ping.
        """
        pytest.skip("Not implemented — requires `make proto` first")

    def test_target_service_ping(self):
        """Target service should respond to health check.

        .. todo::
            Start an in-process Target gRPC server and call Ping.
        """
        pytest.skip("Not implemented — requires `make proto` first")

    def test_full_roundtrip(self):
        """Draft → Target roundtrip should produce accepted tokens.

        .. todo::
            1. Start both services in-process.
            2. Send a DraftRequest to the Draft service.
            3. Forward the DraftResponse to the Target service.
            4. Assert the VerifyResponse contains accepted tokens.
        """
        pytest.skip("Not implemented — requires `make proto` first")
