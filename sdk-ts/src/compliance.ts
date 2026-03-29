/**
 * aria-bsv/compliance — EU AI Act compliance report generator.
 *
 * Produces structured compliance reports referencing Regulation (EU) 2024/1689
 * (EU AI Act) for ARIA-audited AI systems.
 */

// ---------------------------------------------------------------------------
// Enums and interfaces
// ---------------------------------------------------------------------------

/** EU AI Act risk classification tiers (Art. 6 and Annex III). */
export enum RiskTier {
  UNACCEPTABLE = "unacceptable",
  HIGH         = "high",
  LIMITED      = "limited",
  MINIMAL      = "minimal",
}

/** A single epoch's on-chain evidence contributed to a compliance report. */
export interface ComplianceEvidence {
  epochId: string;
  /** BSV (or other chain) transaction ID, if the epoch was anchored on-chain. */
  txid?: string;
  merkleRoot: string;
  recordsCount: number;
  /** ISO-8601 UTC timestamp when the epoch was opened. */
  openedAt: string;
  /** ISO-8601 UTC timestamp when the epoch was closed, if available. */
  closedAt?: string;
  /** SHA-256 hash of the model file committed in this epoch. */
  modelHash: string;
  systemId: string;
}

/** A structured compliance report covering a given audit period. */
export interface ComplianceReport {
  /** Unique report identifier (generated at report creation time). */
  reportId: string;
  /** ISO-8601 UTC timestamp when the report was generated. */
  generatedAt: string;
  systemId: string;
  riskTier: RiskTier;
  /** ISO-8601 start of the audit period (inclusive). */
  auPeriodStart: string;
  /** ISO-8601 end of the audit period (inclusive). */
  auPeriodEnd: string;
  /** Number of evidence items included in this report. */
  evidenceCount: number;
  evidenceItems: ComplianceEvidence[];
  /** True if all mandatory requirements for the risk tier are satisfied. */
  passed: boolean;
  /** Human-readable findings and observations. */
  findings: string[];
  /** EU AI Act article references applicable to this risk tier. */
  articleReferences: string[];
}

// ---------------------------------------------------------------------------
// ComplianceReporter
// ---------------------------------------------------------------------------

/**
 * Generates EU AI Act compliance reports from ARIA audit evidence.
 *
 * @example
 * ```ts
 * const reporter = new ComplianceReporter({
 *   systemId: "my-ai-system",
 *   riskTier: RiskTier.HIGH,
 * });
 * reporter.addEvidence(epochEvidence);
 * const report = reporter.generate("2025-01-01T00:00:00Z", "2025-12-31T23:59:59Z");
 * console.log(reporter.toMarkdown(report));
 * ```
 */
export class ComplianceReporter {
  private readonly _systemId: string;
  private readonly _riskTier: RiskTier;
  private _evidence: ComplianceEvidence[] = [];

  constructor(options: { systemId: string; riskTier: RiskTier }) {
    this._systemId = options.systemId;
    this._riskTier = options.riskTier;
  }

  /** Add an audit evidence item to the reporter's pool. */
  addEvidence(ev: ComplianceEvidence): void {
    this._evidence.push(ev);
  }

  /**
   * Generate a compliance report for the given audit period.
   *
   * Evidence items are filtered to those whose `openedAt` falls within
   * [periodStart, periodEnd].  The report includes EU AI Act article
   * references and findings appropriate to the registered risk tier.
   *
   * @param periodStart ISO-8601 UTC start of the audit period.
   * @param periodEnd   ISO-8601 UTC end of the audit period.
   */
  generate(periodStart: string, periodEnd: string): ComplianceReport {
    const start = new Date(periodStart).getTime();
    const end   = new Date(periodEnd).getTime();

    const items = this._evidence.filter((ev) => {
      const t = new Date(ev.openedAt).getTime();
      return t >= start && t <= end;
    });

    const findings          = _buildFindings(items, this._riskTier);
    const articleReferences = _articleReferencesForTier(this._riskTier);
    const passed            = _evaluatePassed(items, this._riskTier, findings);

    return {
      reportId:      `aria-compliance-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      generatedAt:   new Date().toISOString(),
      systemId:      this._systemId,
      riskTier:      this._riskTier,
      auPeriodStart: periodStart,
      auPeriodEnd:   periodEnd,
      evidenceCount: items.length,
      evidenceItems: items,
      passed,
      findings,
      articleReferences,
    };
  }

  /**
   * Serialise a report to pretty-printed JSON.
   */
  toJson(report: ComplianceReport): string {
    return JSON.stringify(report, null, 2);
  }

  /**
   * Render a report as a structured Markdown document for human review.
   */
  toMarkdown(report: ComplianceReport): string {
    const statusLabel = report.passed ? "PASSED" : "FAILED";

    const lines: string[] = [
      `# ARIA Compliance Report`,
      ``,
      `**Report ID:** ${report.reportId}`,
      `**Generated:** ${report.generatedAt}`,
      `**System:** ${report.systemId}`,
      `**Risk Tier:** ${report.riskTier.toUpperCase()} (EU AI Act, Regulation (EU) 2024/1689)`,
      `**Audit Period:** ${report.auPeriodStart} — ${report.auPeriodEnd}`,
      `**Status:** ${statusLabel}`,
      ``,
      `---`,
      ``,
      `## Evidence Summary`,
      ``,
      `- Total evidence items: **${report.evidenceCount}**`,
    ];

    if (report.evidenceItems.length > 0) {
      lines.push(``, `### Evidence Items`, ``);
      for (const ev of report.evidenceItems) {
        lines.push(
          `#### Epoch \`${ev.epochId}\``,
          `- System: ${ev.systemId}`,
          `- Opened: ${ev.openedAt}`,
          ev.closedAt ? `- Closed: ${ev.closedAt}` : `- Closed: (open)`,
          `- Records: ${ev.recordsCount}`,
          `- Merkle Root: \`${ev.merkleRoot}\``,
          `- Model Hash: \`${ev.modelHash}\``,
          ev.txid
            ? `- On-chain TxID: \`${ev.txid}\``
            : `- On-chain TxID: (not anchored)`,
          ``
        );
      }
    }

    lines.push(`---`, ``, `## Findings`, ``);

    if (report.findings.length === 0) {
      lines.push(`_No findings._`);
    } else {
      for (const f of report.findings) {
        lines.push(`- ${f}`);
      }
    }

    lines.push(
      ``,
      `---`,
      ``,
      `## EU AI Act Article References`,
      ``,
      `_Regulation (EU) 2024/1689_`,
      ``
    );

    for (const ref of report.articleReferences) {
      lines.push(`- ${ref}`);
    }

    lines.push(
      ``,
      `---`,
      ``,
      `## Compliance Decision`,
      ``,
      `> **${statusLabel}**`,
      ``,
      report.passed
        ? `All mandatory requirements for a **${report.riskTier}**-risk AI system have been satisfied for the audit period.`
        : `One or more mandatory requirements for a **${report.riskTier}**-risk AI system have NOT been satisfied. See Findings above.`,
      ``
    );

    return lines.join("\n");
  }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/** Build human-readable findings based on evidence items and risk tier. */
function _buildFindings(
  items: ComplianceEvidence[],
  tier: RiskTier
): string[] {
  const findings: string[] = [];

  if (items.length === 0) {
    findings.push("No audit evidence found for the specified period.");
    return findings;
  }

  // Check all items have a non-zero Merkle root
  const missingRoots = items.filter(
    (ev) => !ev.merkleRoot || ev.merkleRoot === "0".repeat(64)
  );
  if (missingRoots.length > 0) {
    findings.push(
      `${missingRoots.length} evidence item(s) have an empty or zero Merkle root.`
    );
  }

  // Check all items have a valid 64-char model hash
  const missingHashes = items.filter(
    (ev) => !ev.modelHash || ev.modelHash.length !== 64
  );
  if (missingHashes.length > 0) {
    findings.push(
      `${missingHashes.length} evidence item(s) have an invalid or missing model hash.`
    );
  }

  // HIGH and UNACCEPTABLE systems must have on-chain transaction IDs
  if (tier === RiskTier.HIGH || tier === RiskTier.UNACCEPTABLE) {
    const unanchored = items.filter((ev) => !ev.txid);
    if (unanchored.length > 0) {
      findings.push(
        `${unanchored.length} evidence item(s) lack an on-chain transaction ID ` +
        `(required for ${tier}-risk systems under Art. 17).`
      );
    }
  }

  // Epochs with zero records indicate a potential gap in logging
  const emptyEpochs = items.filter((ev) => ev.recordsCount === 0);
  if (emptyEpochs.length > 0) {
    findings.push(
      `${emptyEpochs.length} epoch(s) contain zero records.`
    );
  }

  // Unacceptable tier always warrants an explicit warning
  if (tier === RiskTier.UNACCEPTABLE) {
    findings.push(
      "WARNING: This system is classified as UNACCEPTABLE risk under Art. 5 of the " +
      "EU AI Act. Deployment may be prohibited."
    );
  }

  if (findings.length === 0) {
    findings.push(
      `All ${items.length} evidence item(s) meet the requirements for a ${tier}-risk AI system.`
    );
  }

  return findings;
}

/** Return the EU AI Act article references applicable to the given risk tier. */
function _articleReferencesForTier(tier: RiskTier): string[] {
  switch (tier) {
    case RiskTier.UNACCEPTABLE:
      return [
        "Art. 5 — Prohibited AI practices",
        "Art. 6 — Classification rules for high-risk AI systems",
        "Art. 99 — Penalties for prohibited practices",
      ];
    case RiskTier.HIGH:
      return [
        "Art. 9 — Risk management system",
        "Art. 10 — Data and data governance",
        "Art. 13 — Transparency and provision of information to deployers",
        "Art. 14 — Human oversight",
        "Art. 15 — Accuracy, robustness and cybersecurity",
        "Art. 17 — Quality management system",
        "Art. 26 — Obligations of deployers of high-risk AI systems",
        "Art. 72 — Reporting of serious incidents",
        "Art. 73 — Market surveillance and control",
      ];
    case RiskTier.LIMITED:
      return [
        "Art. 13 — Transparency and provision of information to users",
        "Art. 52 — Transparency obligations for certain AI systems",
      ];
    case RiskTier.MINIMAL:
      return [
        "Art. 95 — Codes of conduct for non-high-risk AI systems",
      ];
  }
}

/** Determine whether the report passes compliance requirements. */
function _evaluatePassed(
  items: ComplianceEvidence[],
  tier: RiskTier,
  findings: string[]
): boolean {
  // No evidence → always fail
  if (items.length === 0) return false;
  // Unacceptable tier → always fail (system must not be deployed)
  if (tier === RiskTier.UNACCEPTABLE) return false;
  // Any finding that indicates a deficiency → fail
  const deficiencyPatterns = [
    "lack", "invalid", "missing", "empty", "zero", "WARNING",
  ];
  const hasDeficiency = findings.some((f) =>
    deficiencyPatterns.some((kw) => f.toLowerCase().includes(kw.toLowerCase()))
  );
  return !hasDeficiency;
}
