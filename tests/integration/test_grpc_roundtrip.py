"""Integration tests for gRPC roundtrip between Draft and Target workers.

Spins up in-process gRPC servers on ephemeral ports with **mocked** engines
so no real model downloads are needed.  Validates the full proto → servicer →
stub transport path for ``DraftService`` and ``TargetService``.

Usage::

    pytest tests/integration/test_grpc_roundtrip.py -v -m integration
"""

from __future__ import annotations

from concurrent import futures
from typing import Any
from unittest.mock import MagicMock

import grpc
import pytest

from specsplit.proto import spec_decoding_pb2, spec_decoding_pb2_grpc
from specsplit.workers.draft.engine import DraftEngine, TokenNode
from specsplit.workers.draft.service import DraftServiceServicer
from specsplit.workers.target.engine import TargetEngine, VerificationResult
from specsplit.workers.target.service import TargetServiceServicer

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Mock engine construction helpers
# ---------------------------------------------------------------------------


def _mock_draft_engine() -> MagicMock:
    """Create a ``DraftEngine`` mock whose ``generate_draft_tree`` returns a
    deterministic two-node chain: ``[TokenNode(42) → TokenNode(43)]``.
    """
    engine = MagicMock(spec=DraftEngine)
    engine.generate_draft_tree.return_value = [
        TokenNode(
            token_id=42,
            log_prob=-0.1,
            children=[
                TokenNode(token_id=43, log_prob=-0.2, children=[]),
            ],
        ),
    ]
    return engine


def _mock_target_engine() -> MagicMock:
    """Create a ``TargetEngine`` mock whose ``verify_draft_tree`` returns a
    result that accepts the first token with a correction on the second.
    """
    engine = MagicMock(spec=TargetEngine)
    engine.verify_draft_tree.return_value = VerificationResult(
        accepted_token_ids=[42],
        correction_token_id=99,
        num_accepted=1,
        cache_hit=False,
    )
    engine.end_session.return_value = True
    engine.active_sessions = 0
    return engine


# ---------------------------------------------------------------------------
# Fixtures — spin up / tear down in-process gRPC servers
# ---------------------------------------------------------------------------


@pytest.fixture()
def draft_server() -> tuple[grpc.Server, int]:
    """Start a ``DraftService`` gRPC server on an ephemeral port.

    Yields ``(server, port)`` and stops the server on teardown.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    servicer = DraftServiceServicer(engine=_mock_draft_engine())
    spec_decoding_pb2_grpc.add_DraftServiceServicer_to_server(servicer, server)
    port: int = server.add_insecure_port("[::]:0")
    server.start()
    yield server, port
    server.stop(grace=0)


@pytest.fixture()
def target_server() -> tuple[grpc.Server, int]:
    """Start a ``TargetService`` gRPC server on an ephemeral port.

    Yields ``(server, port)`` and stops the server on teardown.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    servicer = TargetServiceServicer(engine=_mock_target_engine())
    spec_decoding_pb2_grpc.add_TargetServiceServicer_to_server(servicer, server)
    port: int = server.add_insecure_port("[::]:0")
    server.start()
    yield server, port
    server.stop(grace=0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGRPCRoundtrip:
    """End-to-end gRPC tests for the draft → target pipeline."""

    def test_draft_service_ping(self, draft_server: tuple[grpc.Server, int]) -> None:
        """Draft service should respond to a Ping RPC with status='ok'."""
        _server, port = draft_server
        with grpc.insecure_channel(f"localhost:{port}") as channel:
            stub = spec_decoding_pb2_grpc.DraftServiceStub(channel)
            response: spec_decoding_pb2.PingResponse = stub.Ping(
                spec_decoding_pb2.PingRequest()
            )

        assert response.status == "ok"
        assert response.worker_type == "draft"

    def test_target_service_ping(self, target_server: tuple[grpc.Server, int]) -> None:
        """Target service should respond to a Ping RPC with status='ok'."""
        _server, port = target_server
        with grpc.insecure_channel(f"localhost:{port}") as channel:
            stub = spec_decoding_pb2_grpc.TargetServiceStub(channel)
            response: spec_decoding_pb2.PingResponse = stub.Ping(
                spec_decoding_pb2.PingRequest()
            )

        assert response.status == "ok"
        assert response.worker_type == "target"

    def test_full_roundtrip(
        self,
        draft_server: tuple[grpc.Server, int],
        target_server: tuple[grpc.Server, int],
    ) -> None:
        """Draft → Target roundtrip should produce accepted tokens.

        1. Call ``GenerateDrafts`` on the Draft service.
        2. Forward the resulting ``DraftResponse.draft_tree`` into a
           ``VerifyRequest`` sent to the Target service.
        3. Assert the ``VerifyResponse`` contains accepted tokens.
        """
        _ds, draft_port = draft_server
        _ts, target_port = target_server

        # -- Step 1: Generate drafts -----------------------------------------
        with grpc.insecure_channel(f"localhost:{draft_port}") as draft_channel:
            draft_stub = spec_decoding_pb2_grpc.DraftServiceStub(draft_channel)
            draft_response: spec_decoding_pb2.DraftResponse = (
                draft_stub.GenerateDrafts(
                    spec_decoding_pb2.DraftRequest(
                        request_id="roundtrip-001",
                        prompt_token_ids=[1, 2, 3],
                        max_draft_len=2,
                    )
                )
            )

        assert len(draft_response.draft_tree) > 0, "Draft service returned empty tree"

        # -- Step 2: Forward draft tree to target for verification -----------
        with grpc.insecure_channel(f"localhost:{target_port}") as target_channel:
            target_stub = spec_decoding_pb2_grpc.TargetServiceStub(target_channel)
            verify_response: spec_decoding_pb2.VerifyResponse = (
                target_stub.VerifyDrafts(
                    spec_decoding_pb2.VerifyRequest(
                        request_id="roundtrip-001",
                        prompt_token_ids=[1, 2, 3],
                        draft_tree=draft_response.draft_tree,
                        session_id="sess-roundtrip",
                    )
                )
            )

        # -- Step 3: Assert verification results ----------------------------
        assert verify_response.num_accepted > 0, (
            "Target service accepted zero tokens"
        )
        assert len(verify_response.accepted_token_ids) > 0
        assert verify_response.request_id == "roundtrip-001"
