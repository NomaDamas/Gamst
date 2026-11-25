import unittest
import asyncio
from gamst.crew.transport import JsonlUnixServer, jsonl_call


class Transport(unittest.IsolatedAsyncioTestCase):
    async def test_roundtrip_and_error_bound(self):
        async def handler(request):
            return {"id": request["id"], "ok": True, "request_id": request["request_id"]}

        server = JsonlUnixServer(handler, max_line=1024)
        await server.start()
        try:
            result = await jsonl_call(server.path, {"id": 1})
            self.assertEqual(result["id"], 1)
            self.assertTrue(result["request_id"])
            response = await jsonl_call(server.path, {"data": "x" * 2000})
            self.assertIn("error", response)
        finally:
            await server.close()

    async def test_oversized_line_without_newline_is_rejected(self):
        async def handler(request):
            return {"ok": True}

        server = JsonlUnixServer(handler, max_line=256)
        await server.start()
        reader, writer = await asyncio.open_unix_connection(server.path, limit=1024)
        try:
            writer.write(b"x" * 257)
            await writer.drain()
            response = await reader.readuntil(b"\n")
            self.assertIn(b"line exceeds transport bound", response)
        finally:
            writer.close()
            await writer.wait_closed()
            await server.close()

    def test_transport_line_bound_is_large_enough_for_correlated_errors(self):
        with self.assertRaises(ValueError):
            JsonlUnixServer(lambda request: request, max_line=255)

    def test_transport_timeouts_are_finite_and_bounded(self):
        for value in (0, -1, 10**-400, float("nan"), float("inf"), 601, 10**400):
            with self.assertRaises(ValueError):
                JsonlUnixServer(lambda request: request, handler_timeout_s=value)
            with self.assertRaises(ValueError):
                asyncio.run(jsonl_call("/tmp/unused", {}, timeout_s=value))

        # The documented upper boundary remains valid for both endpoints.
        self.assertEqual(
            JsonlUnixServer(lambda request: request, handler_timeout_s=600).handler_timeout_s,
            600.0,
        )

    async def test_handler_timeout_is_bounded_and_reported(self):
        async def handler(request):
            await asyncio.sleep(0.05)
            return {"ok": True}

        server = JsonlUnixServer(handler, handler_timeout_s=0.01)
        await server.start()
        try:
            response = await jsonl_call(server.path, {"id": 1}, timeout_s=1)
            self.assertIn("error", response)
        finally:
            await server.close()

    async def test_response_correlation_mismatch_is_rejected(self):
        async def handler(request):
            return {"request_id": "different"}

        server = JsonlUnixServer(handler)
        await server.start()
        try:
            result = await jsonl_call(server.path, {"request_id": "expected"})
            self.assertIn("response request_id does not match request", result["error"])
        finally:
            await server.close()

    async def test_duplicate_or_nonfinite_request_is_rejected(self):
        async def handler(request):
            return {"request_id": request["request_id"]}

        server = JsonlUnixServer(handler)
        await server.start()
        try:
            reader, writer = await asyncio.open_unix_connection(server.path)
            writer.write(b'{"request_id":"x","request_id":"y"}\n')
            await writer.drain()
            response = await reader.readuntil(b"\n")
            self.assertIn(b"duplicate JSON field", response)
            writer.close()
            await writer.wait_closed()
        finally:
            await server.close()


if __name__ == "__main__":
    unittest.main()
