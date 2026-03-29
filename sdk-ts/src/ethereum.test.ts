/**
 * ethereum.test.ts — EthAnchor and buildEthAnchorData unit tests.
 *
 * All HTTP calls are intercepted via _setEthFetchImpl so no network is needed.
 */

import { describe, it, expect, beforeEach } from "vitest";
import {
  buildEthAnchorData,
  EthAnchor,
  _setEthFetchImpl,
} from "./ethereum.js";
import type { EthAnchorPayload } from "./ethereum.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const SAMPLE_PAYLOAD: EthAnchorPayload = {
  epochId:      "1700000000000_abcdef",
  merkleRoot:   "a".repeat(64),
  recordsCount: 42,
  chainId:      1,
};

function makeRpcResponse(result?: unknown, error?: { message: string }) {
  const body = error ? { error } : { result };
  return {
    ok: true,
    status: 200,
    json: async () => body,
    text: async () => JSON.stringify(body),
  };
}

function makeNetworkError(message: string) {
  return async () => { throw new Error(message); };
}

// Reset fetch implementation before each test
beforeEach(() => {
  _setEthFetchImpl(async () => makeRpcResponse("0x" + "a".repeat(64)));
});

// ---------------------------------------------------------------------------
// buildEthAnchorData — structure
// ---------------------------------------------------------------------------

describe("buildEthAnchorData — structure", () => {
  it("returns a string starting with 0x", () => {
    const hex = buildEthAnchorData(SAMPLE_PAYLOAD);
    expect(hex.startsWith("0x")).toBe(true);
  });

  it("returns exactly 162 hex chars after 0x (81 bytes)", () => {
    const hex = buildEthAnchorData(SAMPLE_PAYLOAD);
    expect(hex.slice(2).length).toBe(162);
  });

  it("starts with ARIA magic 41524941", () => {
    const hex = buildEthAnchorData(SAMPLE_PAYLOAD);
    expect(hex.slice(2, 10)).toBe("41524941");
  });

  it("has version byte 01 at position 4", () => {
    const hex = buildEthAnchorData(SAMPLE_PAYLOAD);
    expect(hex.slice(10, 12)).toBe("01");
  });

  it("encodes epochId at bytes 5-40 (chars 12-92)", () => {
    const hex = buildEthAnchorData(SAMPLE_PAYLOAD);
    const epochField = Buffer.from(hex.slice(12, 12 + 72), "hex").toString("utf8").replace(/\0+$/, "");
    expect(epochField).toBe(SAMPLE_PAYLOAD.epochId);
  });

  it("encodes merkleRoot at bytes 41-72 (chars 94-158)", () => {
    const hex = buildEthAnchorData(SAMPLE_PAYLOAD);
    const rootField = hex.slice(2 + 82, 2 + 82 + 64);
    expect(rootField).toBe(SAMPLE_PAYLOAD.merkleRoot);
  });

  it("encodes recordsCount as 4-byte BE (value 42 = 0x0000002a)", () => {
    const hex = buildEthAnchorData(SAMPLE_PAYLOAD);
    const countField = hex.slice(2 + 82 + 64, 2 + 82 + 64 + 8);
    expect(countField).toBe("0000002a");
  });

  it("encodes chainId as 4-byte BE (value 1 = 0x00000001)", () => {
    const hex = buildEthAnchorData(SAMPLE_PAYLOAD);
    const chainField = hex.slice(2 + 82 + 64 + 8, 2 + 82 + 64 + 8 + 8);
    expect(chainField).toBe("00000001");
  });

  it("epoch longer than 36 bytes is truncated", () => {
    const longId = "x".repeat(100);
    const hex = buildEthAnchorData({ ...SAMPLE_PAYLOAD, epochId: longId });
    const epochField = Buffer.from(hex.slice(12, 12 + 72), "hex").toString("utf8");
    expect(epochField.length).toBe(36);
  });

  it("epoch shorter than 36 bytes is null-padded", () => {
    const shortId = "ab";
    const hex = buildEthAnchorData({ ...SAMPLE_PAYLOAD, epochId: shortId });
    const epochRaw = hex.slice(12, 12 + 72);
    // First 4 chars = "ab" encoded, rest should be "00"
    expect(epochRaw.slice(0, 4)).toBe("6162"); // 'a'=0x61, 'b'=0x62
    expect(epochRaw.slice(4)).toBe("00".repeat(34));
  });

  it("is deterministic for the same payload", () => {
    expect(buildEthAnchorData(SAMPLE_PAYLOAD)).toBe(buildEthAnchorData(SAMPLE_PAYLOAD));
  });

  it("chainId=137 (Polygon) encodes correctly", () => {
    const hex = buildEthAnchorData({ ...SAMPLE_PAYLOAD, chainId: 137 });
    const chainField = hex.slice(2 + 82 + 64 + 8);
    expect(parseInt(chainField.slice(0, 8), 16)).toBe(137);
  });

  it("merkleRoot without 0x prefix is accepted", () => {
    const root = "b".repeat(64);
    const hex = buildEthAnchorData({ ...SAMPLE_PAYLOAD, merkleRoot: root });
    const rootField = hex.slice(2 + 82, 2 + 82 + 64);
    expect(rootField).toBe(root);
  });
});

// ---------------------------------------------------------------------------
// EthAnchor — sendAnchor
// ---------------------------------------------------------------------------

describe("EthAnchor — sendAnchor", () => {
  it("returns propagated:true with txHash on success", async () => {
    const txHash = "0x" + "c".repeat(64);
    _setEthFetchImpl(async () => makeRpcResponse(txHash));

    const anchor = new EthAnchor({ rpcUrl: "https://rpc.example.com", chainId: 1 });
    const result = await anchor.sendAnchor("deadbeef");

    expect(result.propagated).toBe(true);
    expect(result.txHash).toBe(txHash);
    expect(result.message).toBe("OK");
  });

  it("prepends 0x if rawTxHex does not have it", async () => {
    let capturedBody = "";
    _setEthFetchImpl(async (_url, init) => {
      capturedBody = init?.body ?? "";
      return makeRpcResponse("0x" + "d".repeat(64));
    });

    const anchor = new EthAnchor({ rpcUrl: "https://rpc.example.com" });
    await anchor.sendAnchor("aabbcc");

    const parsed = JSON.parse(capturedBody) as { params: string[] };
    expect(parsed.params[0]).toBe("0xaabbcc");
  });

  it("does not double-prepend 0x if already present", async () => {
    let capturedBody = "";
    _setEthFetchImpl(async (_url, init) => {
      capturedBody = init?.body ?? "";
      return makeRpcResponse("0x" + "e".repeat(64));
    });

    const anchor = new EthAnchor({ rpcUrl: "https://rpc.example.com" });
    await anchor.sendAnchor("0xaabbcc");

    const parsed = JSON.parse(capturedBody) as { params: string[] };
    expect(parsed.params[0]).toBe("0xaabbcc");
  });

  it("uses eth_sendRawTransaction method", async () => {
    let capturedBody = "";
    _setEthFetchImpl(async (_url, init) => {
      capturedBody = init?.body ?? "";
      return makeRpcResponse("0x" + "f".repeat(64));
    });

    const anchor = new EthAnchor({ rpcUrl: "https://rpc.example.com" });
    await anchor.sendAnchor("00");

    const parsed = JSON.parse(capturedBody) as { method: string };
    expect(parsed.method).toBe("eth_sendRawTransaction");
  });

  it("returns propagated:false when RPC returns error", async () => {
    _setEthFetchImpl(async () =>
      makeRpcResponse(undefined, { message: "nonce too low" })
    );

    const anchor = new EthAnchor({ rpcUrl: "https://rpc.example.com" });
    const result = await anchor.sendAnchor("00");

    expect(result.propagated).toBe(false);
    expect(result.message).toContain("nonce too low");
    expect(result.txHash).toBe("");
  });

  it("returns propagated:false and message on network error", async () => {
    _setEthFetchImpl(makeNetworkError("ECONNREFUSED"));

    const anchor = new EthAnchor({ rpcUrl: "https://rpc.example.com" });
    const result = await anchor.sendAnchor("00");

    expect(result.propagated).toBe(false);
    expect(result.message).toContain("ECONNREFUSED");
  });

  it("returns propagated:false when result is empty string", async () => {
    _setEthFetchImpl(async () => makeRpcResponse(""));

    const anchor = new EthAnchor({ rpcUrl: "https://rpc.example.com" });
    const result = await anchor.sendAnchor("00");

    expect(result.propagated).toBe(false);
    expect(result.message).toContain("No txHash");
  });
});

// ---------------------------------------------------------------------------
// EthAnchor — getAnchors
// ---------------------------------------------------------------------------

function makeLogs(dataHexes: string[]) {
  const logs = dataHexes.map((data) => ({ data, blockNumber: "0x1" }));
  return makeRpcResponse(logs);
}

describe("EthAnchor — getAnchors", () => {
  it("returns parsed payloads from valid ARIA logs", async () => {
    const hex = buildEthAnchorData(SAMPLE_PAYLOAD);
    _setEthFetchImpl(async () => makeLogs([hex]));

    const anchor = new EthAnchor({ rpcUrl: "https://rpc.example.com", chainId: 1 });
    const anchors = await anchor.getAnchors("0xContractAddr");

    expect(anchors).toHaveLength(1);
    expect(anchors[0].epochId).toBe(SAMPLE_PAYLOAD.epochId);
    expect(anchors[0].merkleRoot).toBe(SAMPLE_PAYLOAD.merkleRoot);
    expect(anchors[0].recordsCount).toBe(SAMPLE_PAYLOAD.recordsCount);
    expect(anchors[0].chainId).toBe(SAMPLE_PAYLOAD.chainId);
  });

  it("ignores logs that do not have ARIA magic", async () => {
    _setEthFetchImpl(async () => makeLogs(["0x" + "00".repeat(81)]));

    const anchor = new EthAnchor({ rpcUrl: "https://rpc.example.com" });
    const anchors = await anchor.getAnchors("0xContractAddr");

    expect(anchors).toHaveLength(0);
  });

  it("ignores logs that are too short", async () => {
    _setEthFetchImpl(async () => makeLogs(["0x41524941"])); // only 4 bytes

    const anchor = new EthAnchor({ rpcUrl: "https://rpc.example.com" });
    const anchors = await anchor.getAnchors("0xContractAddr");

    expect(anchors).toHaveLength(0);
  });

  it("ignores logs with wrong version byte", async () => {
    const raw = "41524941" + "02" + "00".repeat(76); // version=02 instead of 01
    _setEthFetchImpl(async () => makeLogs(["0x" + raw]));

    const anchor = new EthAnchor({ rpcUrl: "https://rpc.example.com" });
    const anchors = await anchor.getAnchors("0xContractAddr");

    expect(anchors).toHaveLength(0);
  });

  it("returns [] when RPC returns error", async () => {
    _setEthFetchImpl(async () => makeRpcResponse(undefined, { message: "execution reverted" }));

    const anchor = new EthAnchor({ rpcUrl: "https://rpc.example.com" });
    const anchors = await anchor.getAnchors("0xContractAddr");

    expect(anchors).toHaveLength(0);
  });

  it("returns [] on network error", async () => {
    _setEthFetchImpl(makeNetworkError("DNS failed"));

    const anchor = new EthAnchor({ rpcUrl: "https://rpc.example.com" });
    const anchors = await anchor.getAnchors("0xContractAddr");

    expect(anchors).toHaveLength(0);
  });

  it("uses eth_getLogs method", async () => {
    let capturedBody = "";
    _setEthFetchImpl(async (_url, init) => {
      capturedBody = init?.body ?? "";
      return makeRpcResponse([]);
    });

    const anchor = new EthAnchor({ rpcUrl: "https://rpc.example.com" });
    await anchor.getAnchors("0xAddr");

    const parsed = JSON.parse(capturedBody) as { method: string };
    expect(parsed.method).toBe("eth_getLogs");
  });

  it("formats fromBlock as hex", async () => {
    let capturedBody = "";
    _setEthFetchImpl(async (_url, init) => {
      capturedBody = init?.body ?? "";
      return makeRpcResponse([]);
    });

    const anchor = new EthAnchor({ rpcUrl: "https://rpc.example.com" });
    await anchor.getAnchors("0xAddr", 255);

    const parsed = JSON.parse(capturedBody) as { params: Array<{ fromBlock: string }> };
    expect(parsed.params[0].fromBlock).toBe("0xff");
  });

  it("defaults fromBlock to 0x0 when not provided", async () => {
    let capturedBody = "";
    _setEthFetchImpl(async (_url, init) => {
      capturedBody = init?.body ?? "";
      return makeRpcResponse([]);
    });

    const anchor = new EthAnchor({ rpcUrl: "https://rpc.example.com" });
    await anchor.getAnchors("0xAddr");

    const parsed = JSON.parse(capturedBody) as { params: Array<{ fromBlock: string }> };
    expect(parsed.params[0].fromBlock).toBe("0x0");
  });

  it("parses multiple valid logs", async () => {
    const p1 = { ...SAMPLE_PAYLOAD, recordsCount: 1, chainId: 1 };
    const p2 = { ...SAMPLE_PAYLOAD, recordsCount: 99, chainId: 1 };
    _setEthFetchImpl(async () =>
      makeLogs([buildEthAnchorData(p1), buildEthAnchorData(p2)])
    );

    const anchor = new EthAnchor({ rpcUrl: "https://rpc.example.com", chainId: 1 });
    const anchors = await anchor.getAnchors("0xAddr");

    expect(anchors).toHaveLength(2);
    expect(anchors[0].recordsCount).toBe(1);
    expect(anchors[1].recordsCount).toBe(99);
  });
});
