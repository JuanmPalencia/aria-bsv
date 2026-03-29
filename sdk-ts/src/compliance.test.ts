/**
 * compliance.test.ts — ComplianceReporter unit tests.
 */

import { describe, it, expect } from "vitest";
import { ComplianceReporter, RiskTier } from "./compliance.js";
import type { ComplianceEvidence, ComplianceReport } from "./compliance.js";

// ---------------------------------------------------------------------------
// Sample data
// ---------------------------------------------------------------------------

function makeEvidence(overrides: Partial<ComplianceEvidence> = {}): ComplianceEvidence {
  return {
    epochId:      "epoch-1",
    txid:         "a".repeat(64),
    merkleRoot:   "b".repeat(64),
    recordsCount: 10,
    openedAt:     "2025-06-15T12:00:00.000Z",
    closedAt:     "2025-06-15T12:01:00.000Z",
    modelHash:    "c".repeat(64),
    systemId:     "test-system",
    ...overrides,
  };
}

const PERIOD_START = "2025-01-01T00:00:00.000Z";
const PERIOD_END   = "2025-12-31T23:59:59.999Z";

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

describe("ComplianceReporter — constructor", () => {
  it("can be constructed with systemId and riskTier", () => {
    expect(() =>
      new ComplianceReporter({ systemId: "my-system", riskTier: RiskTier.HIGH })
    ).not.toThrow();
  });

  it("accepts all four RiskTier values", () => {
    for (const tier of Object.values(RiskTier)) {
      expect(() =>
        new ComplianceReporter({ systemId: "s", riskTier: tier })
      ).not.toThrow();
    }
  });
});

// ---------------------------------------------------------------------------
// addEvidence + generate — filtering by period
// ---------------------------------------------------------------------------

describe("ComplianceReporter — period filtering", () => {
  it("includes evidence within the period", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.HIGH });
    r.addEvidence(makeEvidence({ openedAt: "2025-06-15T12:00:00.000Z", txid: "a".repeat(64) }));
    const report = r.generate(PERIOD_START, PERIOD_END);
    expect(report.evidenceCount).toBe(1);
  });

  it("excludes evidence before the period", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.HIGH });
    r.addEvidence(makeEvidence({ openedAt: "2024-12-31T23:59:59.999Z", txid: "a".repeat(64) }));
    const report = r.generate(PERIOD_START, PERIOD_END);
    expect(report.evidenceCount).toBe(0);
  });

  it("excludes evidence after the period", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.HIGH });
    r.addEvidence(makeEvidence({ openedAt: "2026-01-01T00:00:00.000Z", txid: "a".repeat(64) }));
    const report = r.generate(PERIOD_START, PERIOD_END);
    expect(report.evidenceCount).toBe(0);
  });

  it("filters correctly when multiple items span different periods", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.LIMITED });
    r.addEvidence(makeEvidence({ epochId: "ep1", openedAt: "2025-03-01T00:00:00.000Z" }));
    r.addEvidence(makeEvidence({ epochId: "ep2", openedAt: "2024-01-01T00:00:00.000Z" }));
    r.addEvidence(makeEvidence({ epochId: "ep3", openedAt: "2025-09-01T00:00:00.000Z" }));
    const report = r.generate(PERIOD_START, PERIOD_END);
    expect(report.evidenceCount).toBe(2);
    const ids = report.evidenceItems.map((e) => e.epochId);
    expect(ids).toContain("ep1");
    expect(ids).toContain("ep3");
  });
});

// ---------------------------------------------------------------------------
// generate() — report metadata
// ---------------------------------------------------------------------------

describe("ComplianceReporter — generate() metadata", () => {
  it("report has a non-empty reportId", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.MINIMAL });
    r.addEvidence(makeEvidence());
    const report = r.generate(PERIOD_START, PERIOD_END);
    expect(report.reportId.length).toBeGreaterThan(0);
  });

  it("reportId starts with aria-compliance-", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.MINIMAL });
    r.addEvidence(makeEvidence());
    const report = r.generate(PERIOD_START, PERIOD_END);
    expect(report.reportId.startsWith("aria-compliance-")).toBe(true);
  });

  it("generatedAt is a recent ISO-8601 timestamp", () => {
    const before = new Date().toISOString();
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.MINIMAL });
    r.addEvidence(makeEvidence());
    const report = r.generate(PERIOD_START, PERIOD_END);
    const after = new Date().toISOString();
    expect(report.generatedAt >= before).toBe(true);
    expect(report.generatedAt <= after).toBe(true);
  });

  it("systemId matches constructor value", () => {
    const r = new ComplianceReporter({ systemId: "my-special-system", riskTier: RiskTier.HIGH });
    r.addEvidence(makeEvidence({ txid: "a".repeat(64) }));
    const report = r.generate(PERIOD_START, PERIOD_END);
    expect(report.systemId).toBe("my-special-system");
  });

  it("riskTier is preserved in report", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.HIGH });
    r.addEvidence(makeEvidence({ txid: "a".repeat(64) }));
    const report = r.generate(PERIOD_START, PERIOD_END);
    expect(report.riskTier).toBe(RiskTier.HIGH);
  });

  it("auPeriodStart and auPeriodEnd are echoed back", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.MINIMAL });
    r.addEvidence(makeEvidence());
    const report = r.generate(PERIOD_START, PERIOD_END);
    expect(report.auPeriodStart).toBe(PERIOD_START);
    expect(report.auPeriodEnd).toBe(PERIOD_END);
  });
});

// ---------------------------------------------------------------------------
// generate() — findings and passed logic
// ---------------------------------------------------------------------------

describe("ComplianceReporter — findings and passed", () => {
  it("passed=false when no evidence in period", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.MINIMAL });
    const report = r.generate(PERIOD_START, PERIOD_END);
    expect(report.passed).toBe(false);
    expect(report.findings.some((f) => f.includes("No audit evidence"))).toBe(true);
  });

  it("passed=false for UNACCEPTABLE tier even with full evidence", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.UNACCEPTABLE });
    r.addEvidence(makeEvidence({ txid: "a".repeat(64) }));
    const report = r.generate(PERIOD_START, PERIOD_END);
    expect(report.passed).toBe(false);
    expect(report.findings.some((f) => f.includes("WARNING"))).toBe(true);
  });

  it("findings include notice about missing txid for HIGH-risk", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.HIGH });
    r.addEvidence(makeEvidence({ txid: undefined })); // no txid
    const report = r.generate(PERIOD_START, PERIOD_END);
    expect(report.findings.some((f) => f.includes("lack"))).toBe(true);
    expect(report.passed).toBe(false);
  });

  it("passed=true for HIGH-risk with all requirements met", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.HIGH });
    r.addEvidence(makeEvidence({ txid: "a".repeat(64), recordsCount: 5 }));
    const report = r.generate(PERIOD_START, PERIOD_END);
    expect(report.passed).toBe(true);
  });

  it("passed=true for LIMITED-risk without txid", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.LIMITED });
    r.addEvidence(makeEvidence({ txid: undefined }));
    const report = r.generate(PERIOD_START, PERIOD_END);
    expect(report.passed).toBe(true);
  });

  it("findings warn about zero-record epochs", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.LIMITED });
    r.addEvidence(makeEvidence({ recordsCount: 0, txid: undefined }));
    const report = r.generate(PERIOD_START, PERIOD_END);
    expect(report.findings.some((f) => f.includes("zero records"))).toBe(true);
  });

  it("findings warn about invalid model hashes", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.LIMITED });
    r.addEvidence(makeEvidence({ modelHash: "tooshort", txid: undefined }));
    const report = r.generate(PERIOD_START, PERIOD_END);
    expect(report.findings.some((f) => f.includes("model hash"))).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// generate() — article references by tier
// ---------------------------------------------------------------------------

describe("ComplianceReporter — article references by tier", () => {
  it("UNACCEPTABLE includes Art. 5", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.UNACCEPTABLE });
    r.addEvidence(makeEvidence());
    const { articleReferences } = r.generate(PERIOD_START, PERIOD_END);
    expect(articleReferences.some((a) => a.includes("Art. 5"))).toBe(true);
  });

  it("HIGH includes Art. 9, 13, 17, and 26", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.HIGH });
    r.addEvidence(makeEvidence({ txid: "a".repeat(64) }));
    const { articleReferences } = r.generate(PERIOD_START, PERIOD_END);
    for (const expected of ["Art. 9", "Art. 13", "Art. 17", "Art. 26"]) {
      expect(articleReferences.some((a) => a.startsWith(expected))).toBe(true);
    }
  });

  it("LIMITED includes Art. 52 and Art. 13", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.LIMITED });
    r.addEvidence(makeEvidence({ txid: undefined }));
    const { articleReferences } = r.generate(PERIOD_START, PERIOD_END);
    expect(articleReferences.some((a) => a.includes("Art. 52"))).toBe(true);
    expect(articleReferences.some((a) => a.startsWith("Art. 13"))).toBe(true);
  });

  it("MINIMAL includes Art. 95", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.MINIMAL });
    r.addEvidence(makeEvidence());
    const { articleReferences } = r.generate(PERIOD_START, PERIOD_END);
    expect(articleReferences.some((a) => a.includes("Art. 95"))).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// toJson()
// ---------------------------------------------------------------------------

describe("ComplianceReporter — toJson()", () => {
  it("returns a valid JSON string", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.MINIMAL });
    r.addEvidence(makeEvidence());
    const report = r.generate(PERIOD_START, PERIOD_END);
    expect(() => JSON.parse(r.toJson(report))).not.toThrow();
  });

  it("serialised JSON round-trips back to the report object", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.MINIMAL });
    r.addEvidence(makeEvidence());
    const report = r.generate(PERIOD_START, PERIOD_END);
    const parsed = JSON.parse(r.toJson(report)) as ComplianceReport;
    expect(parsed.reportId).toBe(report.reportId);
    expect(parsed.systemId).toBe(report.systemId);
    expect(parsed.riskTier).toBe(report.riskTier);
    expect(parsed.evidenceCount).toBe(report.evidenceCount);
  });

  it("is pretty-printed (contains newlines)", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.MINIMAL });
    r.addEvidence(makeEvidence());
    const report = r.generate(PERIOD_START, PERIOD_END);
    expect(r.toJson(report)).toContain("\n");
  });
});

// ---------------------------------------------------------------------------
// toMarkdown()
// ---------------------------------------------------------------------------

describe("ComplianceReporter — toMarkdown()", () => {
  it("returns a string containing the report heading", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.HIGH });
    r.addEvidence(makeEvidence({ txid: "a".repeat(64) }));
    const report = r.generate(PERIOD_START, PERIOD_END);
    const md = r.toMarkdown(report);
    expect(md).toContain("# ARIA Compliance Report");
  });

  it("includes system ID", () => {
    const r = new ComplianceReporter({ systemId: "my-unique-system", riskTier: RiskTier.HIGH });
    r.addEvidence(makeEvidence({ txid: "a".repeat(64) }));
    const report = r.generate(PERIOD_START, PERIOD_END);
    expect(r.toMarkdown(report)).toContain("my-unique-system");
  });

  it("includes PASSED when report passes", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.HIGH });
    r.addEvidence(makeEvidence({ txid: "a".repeat(64) }));
    const report = r.generate(PERIOD_START, PERIOD_END);
    expect(r.toMarkdown(report)).toContain("PASSED");
  });

  it("includes FAILED when report fails", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.HIGH });
    r.addEvidence(makeEvidence({ txid: undefined }));
    const report = r.generate(PERIOD_START, PERIOD_END);
    expect(r.toMarkdown(report)).toContain("FAILED");
  });

  it("includes EU AI Act article references section", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.HIGH });
    r.addEvidence(makeEvidence({ txid: "a".repeat(64) }));
    const report = r.generate(PERIOD_START, PERIOD_END);
    expect(r.toMarkdown(report)).toContain("EU AI Act Article References");
  });

  it("includes each epoch ID in the output", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.LIMITED });
    r.addEvidence(makeEvidence({ epochId: "unique-epoch-xyz", txid: undefined }));
    const report = r.generate(PERIOD_START, PERIOD_END);
    expect(r.toMarkdown(report)).toContain("unique-epoch-xyz");
  });

  it("shows (not anchored) when txid is absent", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.LIMITED });
    r.addEvidence(makeEvidence({ txid: undefined }));
    const report = r.generate(PERIOD_START, PERIOD_END);
    expect(r.toMarkdown(report)).toContain("not anchored");
  });

  it("includes Regulation (EU) 2024/1689 reference", () => {
    const r = new ComplianceReporter({ systemId: "s", riskTier: RiskTier.MINIMAL });
    r.addEvidence(makeEvidence());
    const report = r.generate(PERIOD_START, PERIOD_END);
    expect(r.toMarkdown(report)).toContain("2024/1689");
  });
});
