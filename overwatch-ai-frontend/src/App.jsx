import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

import "./App.css";


/* =========================================================
   COLORS
========================================================= */

const MODULE_A_COLORS = {
  normal: "#54D3A3",
  warning: "#F5C45B",
  anomaly: "#F2757D",
};

const MODULE_B_COLORS = {
  stable: "#62AEEF",
  degrading: "#F27A7F",
};


/* =========================================================
   HELPERS
========================================================= */

function formatNumber(value, digits = 4) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "—";
  }

  return number.toFixed(digits);
}


function getPercentage(value, total) {
  if (!total) {
    return "0.0";
  }

  return (
    (Number(value) / Number(total)) *
    100
  ).toFixed(1);
}


function getRiskInformation(value) {
  const risk = String(value || "")
    .trim()
    .toUpperCase();

  if (
    risk === "HIGH RISK" ||
    risk === "HIGH"
  ) {
    return {
      label: "High Risk",
      className: "high",
    };
  }

  if (
    risk === "MEDIUM RISK" ||
    risk === "MEDIUM" ||
    risk === "NEEDS REVIEW"
  ) {
    return {
      label: "Medium Risk",
      className: "medium",
    };
  }

  return {
    label: "Low Risk",
    className: "low",
  };
}


function getAnomalyInformation(value) {
  const status = String(value || "")
    .trim()
    .toUpperCase();

  if (status === "ANOMALY") {
    return {
      label: "Anomaly",
      className: "anomaly",
    };
  }

  if (status === "WARNING") {
    return {
      label: "Warning",
      className: "warning",
    };
  }

  return {
    label: "Normal",
    className: "normal",
  };
}


function getDegradationInformation(value) {
  const status = String(value || "")
    .trim()
    .toUpperCase();

  if (status === "EARLY REJECTION") {
    return {
      label: "Early Rejection",
      className: "degrading",
    };
  }

  return {
    label: "Acceptable",
    className: "stable",
  };
}


function median(values) {
  const cleanValues = values
    .map(Number)
    .filter(Number.isFinite)
    .sort((a, b) => a - b);

  if (!cleanValues.length) {
    return null;
  }

  const middle = Math.floor(
    cleanValues.length / 2
  );

  if (cleanValues.length % 2 === 0) {
    return (
      cleanValues[middle - 1] +
      cleanValues[middle]
    ) / 2;
  }

  return cleanValues[middle];
}


function calculatePercentageChange(
  startValue,
  endValue
) {
  const start = Number(startValue);
  const end = Number(endValue);

  if (
    !Number.isFinite(start) ||
    !Number.isFinite(end) ||
    start === 0
  ) {
    return null;
  }

  return (
    ((end - start) / Math.abs(start)) *
    100
  );
}


function getRecommendedAction(
  riskClassName,
  degradationClassName
) {
  if (riskClassName === "high") {
    return {
      title: "Early rejection recommended",
      text:
        "This component should not be approved without further QA investigation.",
      className: "high",
    };
  }

  if (riskClassName === "medium") {
    return {
      title: "Manual QA review recommended",
      text:
        "Review this component before it is approved for further use.",
      className: "medium",
    };
  }

  if (degradationClassName === "degrading") {
    return {
      title: "Review predicted degradation",
      text:
        "The component requires additional review because future degradation was detected.",
      className: "medium",
    };
  }

  return {
    title: "Continue normal screening",
    text:
      "No critical reliability concern is currently indicated by the analysis.",
    className: "low",
  };
}


/* =========================================================
   CSV PARSER
========================================================= */

function parseCsvLine(line) {
  const values = [];
  let current = "";
  let insideQuotes = false;

  for (
    let index = 0;
    index < line.length;
    index += 1
  ) {
    const character = line[index];

    if (character === '"') {
      if (
        insideQuotes &&
        line[index + 1] === '"'
      ) {
        current += '"';
        index += 1;
      } else {
        insideQuotes = !insideQuotes;
      }
    } else if (
      character === "," &&
      !insideQuotes
    ) {
      values.push(current.trim());
      current = "";
    } else {
      current += character;
    }
  }

  values.push(current.trim());

  return values;
}


async function readOriginalCsv(file) {
  if (!file) {
    return {};
  }

  const text = await file.text();

  const lines = text
    .replace(/\r/g, "")
    .split("\n")
    .filter(
      (line) => line.trim() !== ""
    );

  if (lines.length < 2) {
    return {};
  }

  const headers = parseCsvLine(
    lines[0]
  ).map((header) =>
    header
      .replace(/^"|"$/g, "")
      .trim()
  );

  const lookup = {};

  for (
    let index = 1;
    index < lines.length;
    index += 1
  ) {
    const values = parseCsvLine(
      lines[index]
    );

    const row = {};

    headers.forEach(
      (header, columnIndex) => {
        row[header] =
          values[columnIndex] ?? "";
      }
    );

    const componentId = String(
      row.Component_ID || ""
    ).trim();

    if (componentId) {
      lookup[componentId] = row;
    }
  }

  return lookup;
}


/* =========================================================
   ANIMATED NUMBER
========================================================= */

function AnimatedNumber({
  value,
  start,
  duration = 800,
}) {
  const [
    displayValue,
    setDisplayValue,
  ] = useState(0);

  useEffect(() => {
    if (!start) {
      setDisplayValue(0);
      return undefined;
    }

    const finalValue =
      Number(value) || 0;

    const startTime =
      performance.now();

    let frame;

    const animate = (time) => {
      const progress = Math.min(
        (time - startTime) /
          duration,
        1
      );

      const eased =
        1 -
        Math.pow(
          1 - progress,
          3
        );

      setDisplayValue(
        Math.round(
          finalValue * eased
        )
      );

      if (progress < 1) {
        frame =
          requestAnimationFrame(
            animate
          );
      }
    };

    frame =
      requestAnimationFrame(
        animate
      );

    return () =>
      cancelAnimationFrame(frame);
  }, [
    value,
    start,
    duration,
  ]);

  return displayValue.toLocaleString();
}


/* =========================================================
   OVERWATCH EXPLAIN
========================================================= */

function OverWatchExplain({
  component,
  batchMedian,
}) {
  if (!component) {
    return null;
  }

  const risk =
    getRiskInformation(
      component.Final_Risk
    );

  const anomaly =
    getAnomalyInformation(
      component.Module_A_Risk
    );

  const degradation =
    getDegradationInformation(
      component.Module_B_Decision
    );

  const initialReading =
    Number(component.Value_0h_uA);

  const reading24 =
    Number(component.Value_24h_uA);

  const predicted168 =
    Number(component.Predicted_168h);

  const driftRate =
    Number(component.Drift_Rate);

  const safetyThreshold =
    Number(
      component.Safety_Threshold
    );

  const batchDeviation =
    Number.isFinite(initialReading) &&
    Number.isFinite(batchMedian) &&
    batchMedian !== 0
      ? (
          ((initialReading -
            batchMedian) /
            Math.abs(batchMedian)) *
          100
        )
      : null;

  const earlyChange =
    calculatePercentageChange(
      initialReading,
      reading24
    );

  const thresholdMultiple =
    Number.isFinite(driftRate) &&
    Number.isFinite(
      safetyThreshold
    ) &&
    safetyThreshold !== 0
      ? driftRate /
        safetyThreshold
      : null;

  const recommendation =
    getRecommendedAction(
      risk.className,
      degradation.className
    );

  const batchExplanation =
    anomaly.className === "anomaly"
      ? "This component behaves very differently from the typical component in its batch."
      : anomaly.className ===
          "warning"
        ? "This component shows some unusual behavior compared with its batch and deserves attention."
        : "This component is currently behaving similarly to the normal pattern of its batch.";

  const predictionExplanation =
    degradation.className ===
    "degrading"
      ? "The early readings suggest that this component may degrade too quickly before the end of Burn-In testing."
      : "The predicted future degradation remains within the acceptable range used by the system.";

  return (
    <section className="overwatch-explain">

      <div className="explain-header">

        <div className="explain-brand-icon">
          <svg
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <path d="M12 3C7.5 3 3.7 6 2 10.5C3.7 15 7.5 18 12 18C16.5 18 20.3 15 22 10.5C20.3 6 16.5 3 12 3Z" />
            <circle
              cx="12"
              cy="10.5"
              r="3"
            />
            <path d="M12 18V21" />
          </svg>
        </div>

        <div>
          <span className="explain-eyebrow">
            OVERWATCH EXPLAIN
          </span>

          <h4>
            Why was this component
            classified this way?
          </h4>

          <p>
            A simple explanation of
            the AI result.
          </p>
        </div>

      </div>


      <div className="explain-list">

        <article className="explain-item">

          <div
            className={`explain-number explain-number-${anomaly.className}`}
          >
            1
          </div>

          <div>
            <h5>
              Compared with its batch
            </h5>

            <p>
              {batchExplanation}
            </p>

            <div className="explain-metrics">

              <span>
                Component
                <strong>
                  {formatNumber(
                    initialReading,
                    2
                  )}{" "}
                  µA
                </strong>
              </span>

              <span>
                Batch typical
                <strong>
                  {formatNumber(
                    batchMedian,
                    2
                  )}{" "}
                  µA
                </strong>
              </span>

              {Number.isFinite(
                batchDeviation
              ) && (
                <span>
                  Difference
                  <strong>
                    {batchDeviation >= 0
                      ? "+"
                      : ""}
                    {formatNumber(
                      batchDeviation,
                      1
                    )}
                    %
                  </strong>
                </span>
              )}

            </div>

          </div>

        </article>


        <article className="explain-item">

          <div className="explain-number">
            2
          </div>

          <div>
            <h5>
              Early change
            </h5>

            <p>
              The first 24 hours show
              how quickly the component
              begins changing during the
              test.
            </p>

            <div className="explain-metrics">

              <span>
                0h
                <strong>
                  {formatNumber(
                    initialReading,
                    2
                  )}{" "}
                  µA
                </strong>
              </span>

              <span>
                24h
                <strong>
                  {formatNumber(
                    reading24,
                    2
                  )}{" "}
                  µA
                </strong>
              </span>

              {Number.isFinite(
                earlyChange
              ) && (
                <span>
                  Change
                  <strong>
                    {earlyChange >= 0
                      ? "+"
                      : ""}
                    {formatNumber(
                      earlyChange,
                      1
                    )}
                    %
                  </strong>
                </span>
              )}

            </div>

          </div>

        </article>


        <article className="explain-item">

          <div
            className={`explain-number explain-number-${degradation.className}`}
          >
            3
          </div>

          <div>
            <h5>
              Future degradation
            </h5>

            <p>
              {predictionExplanation}
            </p>

            <div className="explain-metrics">

              <span>
                Predicted 168h
                <strong>
                  {formatNumber(
                    predicted168,
                    2
                  )}{" "}
                  µA
                </strong>
              </span>

              <span>
                Predicted drift
                <strong>
                  {formatNumber(
                    driftRate,
                    4
                  )}{" "}
                  µA/hr
                </strong>
              </span>

              <span>
                Safe drift threshold
                <strong>
                  {formatNumber(
                    safetyThreshold,
                    4
                  )}{" "}
                  µA/hr
                </strong>
              </span>

            </div>

            {Number.isFinite(
              thresholdMultiple
            ) &&
              degradation.className ===
                "degrading" && (
                <div className="threshold-warning">
                  Predicted degradation
                  is{" "}
                  <strong>
                    {formatNumber(
                      thresholdMultiple,
                      1
                    )}
                    ×
                  </strong>{" "}
                  the calculated safety
                  threshold.
                </div>
              )}

          </div>

        </article>


        <article className="explain-item">

          <div
            className={`explain-number explain-number-${risk.className}`}
          >
            4
          </div>

          <div>
            <h5>
              Final reliability decision
            </h5>

            <p>
              OverWatch combines the
              anomaly result and the
              degradation prediction to
              produce the final risk.
            </p>

            <div className="explain-decision-flow">

              <span
                className={`explain-decision-pill anomaly-${anomaly.className}`}
              >
                {anomaly.label}
              </span>

              <span className="decision-plus">
                +
              </span>

              <span
                className={`explain-decision-pill degradation-${degradation.className}`}
              >
                {degradation.label}
              </span>

              <span className="decision-arrow">
                →
              </span>

              <span
                className={`explain-decision-pill final-${risk.className}`}
              >
                {risk.label}
              </span>

            </div>

          </div>

        </article>

      </div>


      <div
        className={`explain-recommendation explain-recommendation-${recommendation.className}`}
      >

        <div className="recommendation-icon">
          {recommendation.className ===
          "low"
            ? "✓"
            : recommendation.className ===
                "medium"
              ? "!"
              : "×"}
        </div>

        <div>
          <span>
            RECOMMENDED ACTION
          </span>

          <strong>
            {recommendation.title}
          </strong>

          <p>
            {recommendation.text}
          </p>
        </div>

      </div>

    </section>
  );
}


/* =========================================================
   COMPONENT DETAILS PANEL
========================================================= */

function ComponentDetailsPanel({
  component,
  batchMedian,
  onClose,
}) {
  if (!component) {
    return null;
  }

  const finalRisk =
    getRiskInformation(
      component.Final_Risk
    );

  const anomaly =
    getAnomalyInformation(
      component.Module_A_Risk
    );

  const degradation =
    getDegradationInformation(
      component.Module_B_Decision
    );

  const assessmentText = {
    high:
      "Both reliability checks indicate that this component requires immediate attention.",

    medium:
      "At least one reliability check has identified behavior that should be reviewed.",

    low:
      "The current analysis does not indicate a major reliability concern.",
  };

  const reading0 = Number(component.Value_0h_uA);
  const reading24 = Number(component.Value_24h_uA);
  const reading96 = Number(component.Value_96h_uA);
  const reading168 = Number(component.Value_168h_uA);
  const predicted168 = Number(component.Predicted_168h);
  const safetyThreshold = Number(component.Safety_Threshold);

  const safetyAt96 =
    Number.isFinite(reading24) &&
    Number.isFinite(safetyThreshold)
      ? reading24 + safetyThreshold * (96 - 24)
      : null;

  const safetyAt168 =
    Number.isFinite(reading24) &&
    Number.isFinite(safetyThreshold)
      ? reading24 + safetyThreshold * (168 - 24)
      : null;

  const degradationChartData = [
    {
      time: "0h",
      actual: Number.isFinite(reading0) ? reading0 : null,
      prediction: null,
      safety: null,
    },
    {
      time: "24h",
      actual: Number.isFinite(reading24) ? reading24 : null,
      prediction: Number.isFinite(reading24) ? reading24 : null,
      safety: Number.isFinite(reading24) ? reading24 : null,
    },
    {
      time: "96h",
      actual: Number.isFinite(reading96) ? reading96 : null,
      prediction: null,
      safety: Number.isFinite(safetyAt96) ? safetyAt96 : null,
    },
    {
      time: "168h",
      actual: Number.isFinite(reading168) ? reading168 : null,
      prediction: Number.isFinite(predicted168) ? predicted168 : null,
      safety: Number.isFinite(safetyAt168) ? safetyAt168 : null,
    },
  ];

  return (
    <aside className="component-details-panel">

      <div className="details-panel-header">

        <div className="details-panel-title">

          <span className="details-header-icon">
            <svg viewBox="0 0 24 24">
              <path d="M6 3H15L19 7V21H6Z" />
              <path d="M15 3V8H19" />
              <path d="M9 12H16" />
              <path d="M9 16H16" />
            </svg>
          </span>

          <strong>
            Component Details
          </strong>

        </div>

        <button
          type="button"
          className="details-close-button"
          onClick={onClose}
          aria-label="Close component details"
        >
          ×
        </button>

      </div>


      <div className="details-panel-content">

        <div className="details-component-heading">

          <div className="details-component-left">

            <strong>
              {component.Component_ID ||
                "—"}
            </strong>

            <span
              className={`details-risk-badge details-risk-${finalRisk.className}`}
            >
              {finalRisk.label}
            </span>

          </div>

          <div className="details-batch">
            <span>Batch:</span>

            <strong>
              {component.Batch_ID ||
                "—"}
            </strong>
          </div>

        </div>


        <div className="details-section-title">
          <strong>
            Test Readings
          </strong>

          <span
            className="details-info-icon"
            title="Measured leakage current values from the uploaded CSV."
          >
            i
          </span>
        </div>


        <div className="details-reading-list">

          <div className="details-reading-row">
            <span>Initial (0h)</span>
            <strong>
              {formatNumber(
                component.Value_0h_uA
              )}{" "}
              µA
            </strong>
          </div>

          <div className="details-reading-row">
            <span>After 24h</span>
            <strong>
              {formatNumber(
                component.Value_24h_uA
              )}{" "}
              µA
            </strong>
          </div>

          <div className="details-reading-row">
            <span>After 96h</span>
            <strong>
              {formatNumber(
                component.Value_96h_uA
              )}{" "}
              µA
            </strong>
          </div>

          <div className="details-reading-row">
            <span>After 168h</span>
            <strong>
              {formatNumber(
                component.Value_168h_uA
              )}{" "}
              µA
            </strong>
          </div>

        </div>


        <div
          className={`predicted-reading-box predicted-reading-${finalRisk.className}`}
        >
          <span>
            Predicted 168h Reading
          </span>

          <strong>
            {formatNumber(
              component.Predicted_168h
            )}{" "}
            µA
          </strong>
        </div>


        <div className="details-secondary-readings">

          <div className="details-reading-row">
            <span>
              Predicted Drift
            </span>

            <strong>
              {formatNumber(
                component.Drift_Rate,
                5
              )}{" "}
              µA/hr
            </strong>
          </div>

          <div className="details-reading-row">
            <span>
              Safe Drift Threshold
            </span>

            <strong>
              {formatNumber(
                component.Safety_Threshold,
                5
              )}{" "}
              µA/hr
            </strong>
          </div>

        </div>


        <section className="component-degradation-chart">

          <div className="degradation-chart-heading">
            <div>
              <strong>
                Component Degradation Timeline
              </strong>
              <span>
                Actual readings, predicted 168h value and calculated safe drift boundary
              </span>
            </div>

            <span className="degradation-chart-unit">
              µA
            </span>
          </div>

          <div className="degradation-chart-canvas">

  <ResponsiveContainer width="100%" height={300}>

    <LineChart
      data={degradationChartData}
      margin={{
        top: 15,
        right: 25,
        left: 5,
        bottom: 10,
      }}
    >

      <CartesianGrid
        strokeDasharray="3 3"
        vertical={false}
      />

      <XAxis
        dataKey="time"
        tickLine={false}
        axisLine={false}
        tickMargin={10}
      />

      <YAxis
        tickLine={false}
        axisLine={false}
        tickMargin={8}
        width={45}
        domain={["auto", "auto"]}
      />

      <Tooltip
        formatter={(value, name) => [
          `${Number(value).toFixed(2)} µA`,
          name,
        ]}
      />

      <Legend
        verticalAlign="bottom"
        height={32}
        iconType="line"
      />

      <Line
        type="linear"
        dataKey="actual"
        name="Actual Reading"
        stroke="#8FB7FF"
        strokeWidth={3}
        dot={{
          r: 4,
          fill: "#8FB7FF",
          stroke: "#8FB7FF",
          strokeWidth: 1,
        }}
        activeDot={{
          r: 6,
        }}
        connectNulls
        isAnimationActive={false}
      />

      <Line
        type="linear"
        dataKey="prediction"
        name="Predicted Reading"
        stroke="#A78BFA"
        strokeWidth={3}
        strokeDasharray="7 5"
        dot={{
          r: 4,
          fill: "#A78BFA",
          stroke: "#A78BFA",
          strokeWidth: 1,
        }}
        connectNulls
        isAnimationActive={false}
      />

      <Line
        type="linear"
        dataKey="safety"
        name="Safe Drift Boundary"
        stroke="#F5C45B"
        strokeWidth={2.5}
        strokeDasharray="5 5"
        dot={false}
        connectNulls
        isAnimationActive={false}
      />

    </LineChart>

  </ResponsiveContainer>

</div>

          <div className="degradation-chart-note">
            <span className="degradation-chart-note-dot" />
            <span>
              Safe boundary is calculated from the 24h reading using the model&apos;s
              safe drift threshold of{" "}
              <strong>
                {formatNumber(
                  component.Safety_Threshold,
                  5
                )}{" "}
                µA/hr
              </strong>.
            </span>
          </div>

        </section>


        <div className="details-modules-grid">

          <div className="details-module-card">

            <div className="details-module-heading">

              <span className="details-module-icon module-a-mini-icon">
                <svg viewBox="0 0 24 24">
                  <path d="M3 13H7L9 7L12 18L15 10L17 13H21" />
                </svg>
              </span>

              <div>
                <strong>
                  Module A — Anomaly Detection
                </strong>

                <span>
                  Compared with its batch
                </span>
              </div>

            </div>

            <div className="details-module-values">

              <div>
                <span>Status</span>

                <strong
                  className={`details-text-${anomaly.className}`}
                >
                  {anomaly.label}
                </strong>
              </div>

              <div>
                <span>
                  Anomaly Score
                </span>

                <strong>
                  {formatNumber(
                    component
                      .Dynamic_Anomaly_Score,
                    2
                  )}
                </strong>
              </div>

            </div>

          </div>


          <div className="details-module-card">

            <div className="details-module-heading">

              <span className="details-module-icon module-b-mini-icon">
                <svg viewBox="0 0 24 24">
                  <path d="M5 20V14" />
                  <path d="M10 20V10" />
                  <path d="M15 20V7" />
                  <path d="M20 20V4" />
                  <path d="M4 10L9 7L14 5L20 2" />
                </svg>
              </span>

              <div>
                <strong>
                  Module B — Degradation Prediction
                </strong>

                <span>
                  Predicted degradation
                </span>
              </div>

            </div>

            <div className="details-module-values">

              <div>
                <span>Status</span>

                <strong
                  className={`details-text-${degradation.className}`}
                >
                  {degradation.label}
                </strong>
              </div>

              <div>
                <span>
                  Predicted 168h
                </span>

                <strong>
                  {formatNumber(
                    component.Predicted_168h
                  )}{" "}
                  µA
                </strong>
              </div>

            </div>

          </div>

        </div>


        <div
          className={`details-final-assessment details-final-${finalRisk.className}`}
        >

          <span
            className={`details-final-icon details-final-icon-${finalRisk.className}`}
          >
            {finalRisk.className ===
            "low"
              ? "✓"
              : finalRisk.className ===
                  "medium"
                ? "▲"
                : "!"}
          </span>

          <div>
            <strong>
              Final Reliability Assessment:{" "}
              {finalRisk.label}
            </strong>

            <p>
              {
                assessmentText[
                  finalRisk.className
                ]
              }
            </p>
          </div>

        </div>


        <OverWatchExplain
          component={component}
          batchMedian={batchMedian}
        />

      </div>

    </aside>
  );
}


/* =========================================================
   BATCH HEALTH INTELLIGENCE
========================================================= */

function BatchHealthIntelligence({
  batches,
  selectedBatch,
  onSelectBatch,
}) {
  if (!batches.length) {
    return null;
  }

  const batch =
    batches.find(
      (item) =>
        item.batchId === selectedBatch
    ) || batches[0];

  const healthClass =
    batch.highRate >= 15
      ? "critical"
      : batch.highRate >= 5 ||
          batch.mediumRate >= 20
        ? "watch"
        : "healthy";

  const healthLabel =
    healthClass === "critical"
      ? "Needs Investigation"
      : healthClass === "watch"
        ? "Watch Closely"
        : "Healthy Batch";

  const healthDescription =
    healthClass === "critical"
      ? "This batch contains an unusually high concentration of high-risk components."
      : healthClass === "watch"
        ? "This batch contains enough unusual components to deserve closer QA attention."
        : "Most components in this batch currently show acceptable reliability behavior.";

  return (
    <section className="batch-intelligence-section">

      <div className="batch-intelligence-heading">

        <div>
          <span className="dashboard-eyebrow">
            BATCH-LEVEL INSIGHT
          </span>

          <h2>
            Batch Health Intelligence
          </h2>

          <p>
            See whether reliability
            problems are isolated to one
            component or concentrated
            inside a manufacturing batch.
          </p>
        </div>

        <div className="batch-intelligence-summary">
          <span>
            {batches.length}
          </span>
          <small>
            batches analyzed
          </small>
        </div>

      </div>


      <div className="batch-intelligence-layout">

        <div className="batch-list-card">

          <div className="batch-list-header">
            <strong>
              Batch Overview
            </strong>

            <span>
              Select a batch to inspect
            </span>
          </div>

          <div className="batch-list">

            {batches.map((item) => {

              const itemHealthClass =
                item.highRate >= 15
                  ? "critical"
                  : item.highRate >= 5 ||
                      item.mediumRate >=
                        20
                    ? "watch"
                    : "healthy";

              return (
                <button
                  type="button"
                  key={item.batchId}
                  className={`batch-list-item ${
                    batch.batchId ===
                    item.batchId
                      ? "active"
                      : ""
                  }`}
                  onClick={() =>
                    onSelectBatch(
                      item.batchId
                    )
                  }
                >

                  <div className="batch-list-name">
                    <span
                      className={`batch-health-dot batch-health-${itemHealthClass}`}
                    />

                    <div>
                      <strong>
                        {item.batchId}
                      </strong>

                      <small>
                        {item.total}{" "}
                        components
                      </small>
                    </div>
                  </div>

                  <div className="batch-list-risk">
                    <strong>
                      {formatNumber(
                        item.highRate,
                        1
                      )}
                      %
                    </strong>

                    <span>
                      high risk
                    </span>
                  </div>

                </button>
              );
            })}

          </div>

        </div>


        <article className="batch-detail-card">

          <div className="batch-detail-top">

            <div>
              <span className="batch-detail-eyebrow">
                SELECTED BATCH
              </span>

              <h3>
                {batch.batchId}
              </h3>

              <p>
                {healthDescription}
              </p>
            </div>

            <span
              className={`batch-health-badge batch-health-${healthClass}`}
            >
              {healthLabel}
            </span>

          </div>


          <div className="batch-stat-grid">

            <div className="batch-stat">
              <span>
                Components
              </span>

              <strong>
                {batch.total}
              </strong>
            </div>

            <div className="batch-stat batch-stat-low">
              <span>
                Low Risk
              </span>

              <strong>
                {batch.low}
              </strong>
            </div>

            <div className="batch-stat batch-stat-medium">
              <span>
                Need Review
              </span>

              <strong>
                {batch.medium}
              </strong>
            </div>

            <div className="batch-stat batch-stat-high">
              <span>
                High Risk
              </span>

              <strong>
                {batch.high}
              </strong>
            </div>

          </div>


          <div className="batch-insight-grid">

            <div className="batch-insight">

              <div className="batch-insight-icon">
                ∿
              </div>

              <div>
                <span>
                  Anomaly concentration
                </span>

                <strong>
                  {formatNumber(
                    batch.anomalyRate,
                    1
                  )}
                  %
                </strong>

                <small>
                  {batch.anomaly} of{" "}
                  {batch.total} components
                </small>
              </div>

            </div>


            <div className="batch-insight">

              <div className="batch-insight-icon">
                ↗
              </div>

              <div>
                <span>
                  Early rejection rate
                </span>

                <strong>
                  {formatNumber(
                    batch.earlyRejectionRate,
                    1
                  )}
                  %
                </strong>

                <small>
                  {
                    batch.earlyRejection
                  }{" "}
                  of {batch.total} components
                </small>
              </div>

            </div>


            <div className="batch-insight">

              <div className="batch-insight-icon">
                ◉
              </div>

              <div>
                <span>
                  Typical initial reading
                </span>

                <strong>
                  {formatNumber(
                    batch.median0h,
                    2
                  )}{" "}
                  µA
                </strong>

                <small>
                  Median for this batch
                </small>
              </div>

            </div>

          </div>


          <div
            className={`batch-recommendation batch-recommendation-${healthClass}`}
          >
            <strong>
              {healthClass ===
              "critical"
                ? "Investigate this batch"
                : healthClass ===
                    "watch"
                  ? "Keep this batch under review"
                  : "No batch-level warning detected"}
            </strong>

            <p>
              {healthClass ===
              "critical"
                ? "Several components from the same batch are showing serious reliability concerns. Review the batch before approving the remaining components."
                : healthClass ===
                    "watch"
                  ? "Some components in this batch show unusual behavior. A QA review is recommended before final approval."
                  : "The batch does not currently show a strong concentration of high-risk components."}
            </p>
          </div>

        </article>

      </div>

    </section>
  );
}


/* =========================================================
   MAIN APP
========================================================= */

export default function App() {

  const [file, setFile] =
    useState(null);

  const [darkMode, setDarkMode] =
    useState(false);

  const [loading, setLoading] =
    useState(false);

  const [results, setResults] =
    useState(null);

  const [error, setError] =
    useState("");

  const [
    originalCsvLookup,
    setOriginalCsvLookup,
  ] = useState({});

  const [
    selectedComponent,
    setSelectedComponent,
  ] = useState(null);

  const [
    selectedBatch,
    setSelectedBatch,
  ] = useState(null);

  const [search, setSearch] =
    useState("");

  const [riskFilter, setRiskFilter] =
    useState("all");

  const [
    currentPage,
    setCurrentPage,
  ] = useState(1);

  const [
    rowsPerPage,
    setRowsPerPage,
  ] = useState(5);

  const [pageJump, setPageJump] =
    useState("1");

  const [
    showResultsAnimation,
    setShowResultsAnimation,
  ] = useState(false);

  const fileInputRef =
    useRef(null);


  /* =======================================================
     DARK MODE
  ======================================================= */

  useEffect(() => {
    document.documentElement.classList.toggle(
      "dark-mode",
      darkMode
    );

    return () => {
      document.documentElement.classList.remove(
        "dark-mode"
      );
    };
  }, [darkMode]);


  /* =======================================================
     FILE
  ======================================================= */

  const resetResults = () => {
    setResults(null);
    setError("");
    setOriginalCsvLookup({});
    setSelectedComponent(null);
    setSelectedBatch(null);
    setSearch("");
    setRiskFilter("all");
    setCurrentPage(1);
    setPageJump("1");
    setShowResultsAnimation(false);
  };


  const handleFile = (
    selectedFile
  ) => {
    if (!selectedFile) {
      return;
    }

    if (
      !selectedFile.name
        .toLowerCase()
        .endsWith(".csv")
    ) {
      alert(
        "Only CSV files are allowed."
      );

      return;
    }

    setFile(selectedFile);
    resetResults();
  };


  const removeFile = (event) => {
    event.stopPropagation();

    setFile(null);
    resetResults();

    if (fileInputRef.current) {
      fileInputRef.current.value =
        "";
    }
  };


  /* =======================================================
     ANALYSIS
  ======================================================= */

  const handleAnalysis = async () => {
  if (!file || loading) {
    return;
  }

  setLoading(true);
  setError("");
  setSelectedComponent(null);
  setSelectedBatch(null);

  try {
    const csvLookup = await readOriginalCsv(file);

    // Validate that the CSV actually contains usable component data
    const csvRows = Object.values(csvLookup);

    if (csvRows.length === 0) {
      throw new Error(
        "INVALID_CSV: The CSV is empty or does not contain a valid Component_ID column."
      );
    }

    const requiredColumns = [
      "Component_ID",
      "Batch_ID",
      "Value_0h_uA",
      "Value_24h_uA",
    ];

    const firstRow = csvRows[0];

    const missingColumns = requiredColumns.filter(
      (column) =>
        !Object.prototype.hasOwnProperty.call(firstRow, column)
    );

    if (missingColumns.length > 0) {
      throw new Error(
        `INVALID_CSV: Missing required columns: ${missingColumns.join(", ")}`
      );
    }

    setOriginalCsvLookup(csvLookup);

    const formData = new FormData();

    formData.append("file", file);

    const response = await fetch(
      "http://localhost:8080/api/ai/risk-analysis",
      {
        method: "POST",
        body: formData,
      }
    );

    if (!response.ok) {
      let backendMessage = "";

      try {
        const errorData = await response.json();

        backendMessage =
          errorData?.message ||
          errorData?.detail ||
          errorData?.error ||
          "";
      } catch {
        try {
          backendMessage = await response.text();
        } catch {
          backendMessage = "";
        }
      }

      throw new Error(
        backendMessage
          ? `BACKEND_ERROR: ${backendMessage}`
          : `BACKEND_ERROR: Analysis failed with status ${response.status}`
      );
    }

    const data = await response.json();

    setResults(data);

    setTimeout(() => {
      setShowResultsAnimation(true);
    }, 50);
  } catch (analysisError) {
    console.error(analysisError);

    const message =
      analysisError instanceof Error
        ? analysisError.message
        : "";

    if (message.startsWith("INVALID_CSV:")) {
      setError(
        message.replace("INVALID_CSV:", "").trim()
      );
    } else if (message.startsWith("BACKEND_ERROR:")) {
      setError(
        message.replace("BACKEND_ERROR:", "").trim()
      );
    } else if (
      analysisError instanceof TypeError &&
      message.toLowerCase().includes("fetch")
    ) {
      setError(
        "Unable to connect to the OverWatch backend. Please verify that the Spring Boot and AI services are running."
      );
    } else {
      setError(
        "Analysis could not be completed. Please check the uploaded dataset and try again."
      );
    }
  } finally {
    setLoading(false);
  }
};


  /* =======================================================
     MERGE BACKEND + ORIGINAL CSV
  ======================================================= */

  const components =
    useMemo(() => {

      if (
        !Array.isArray(
          results?.results
        )
      ) {
        return [];
      }

      const riskPriority = {
        "HIGH RISK": 1,
        "MEDIUM RISK": 2,
        "LOW RISK": 3,
      };

      return results.results
        .map((component) => {

          const componentId =
            String(
              component.Component_ID ||
                ""
            ).trim();

          const original =
            originalCsvLookup[
              componentId
            ] || {};

          return {
            ...original,
            ...component,

            Value_0h_uA:
              component.Value_0h_uA ??
              original.Value_0h_uA,

            Value_24h_uA:
              component.Value_24h_uA ??
              original.Value_24h_uA,

            Value_96h_uA:
              component.Value_96h_uA ??
              original.Value_96h_uA,

            Value_168h_uA:
              component.Value_168h_uA ??
              original.Value_168h_uA,
          };
        })
        .sort((a, b) => {

          const aRisk =
            riskPriority[
              String(
                a.Final_Risk || ""
              )
                .trim()
                .toUpperCase()
            ] ?? 4;

          const bRisk =
            riskPriority[
              String(
                b.Final_Risk || ""
              )
                .trim()
                .toUpperCase()
            ] ?? 4;

          return aRisk - bRisk;
        });

    }, [
      results,
      originalCsvLookup,
    ]);


  /* =======================================================
     MODULE A
  ======================================================= */

  const moduleAStats =
    useMemo(() => {

      const stats = {
        normal: 0,
        warning: 0,
        anomaly: 0,
      };

      components.forEach(
        (component) => {

          const risk =
            String(
              component.Module_A_Risk ||
                ""
            )
              .trim()
              .toUpperCase();

          if (risk === "ANOMALY") {
            stats.anomaly += 1;
          } else if (
            risk === "WARNING"
          ) {
            stats.warning += 1;
          } else if (
            risk === "NORMAL"
          ) {
            stats.normal += 1;
          }
        }
      );

      return stats;

    }, [components]);


  const moduleATotal =
    moduleAStats.normal +
    moduleAStats.warning +
    moduleAStats.anomaly;


  const moduleAData = [
    {
      name: "Normal",
      value: moduleAStats.normal,
      color:
        MODULE_A_COLORS.normal,
    },
    {
      name: "Warning",
      value: moduleAStats.warning,
      color:
        MODULE_A_COLORS.warning,
    },
    {
      name: "Anomaly",
      value: moduleAStats.anomaly,
      color:
        MODULE_A_COLORS.anomaly,
    },
  ];


  /* =======================================================
     MODULE B
  ======================================================= */

  const moduleBStats =
    useMemo(() => {

      const stats = {
        stable: 0,
        degrading: 0,
      };

      components.forEach(
        (component) => {

          const decision =
            String(
              component
                .Module_B_Decision ||
                ""
            )
              .trim()
              .toUpperCase();

          if (
            decision ===
            "EARLY REJECTION"
          ) {
            stats.degrading += 1;
          } else if (
            decision ===
            "ACCEPTABLE"
          ) {
            stats.stable += 1;
          }
        }
      );

      return stats;

    }, [components]);


  const moduleBTotal =
    moduleBStats.stable +
    moduleBStats.degrading;


  const moduleBData = [
    {
      name: "Acceptable",
      value: moduleBStats.stable,
      color:
        MODULE_B_COLORS.stable,
    },
    {
      name: "Early Rejection",
      value:
        moduleBStats.degrading,
      color:
        MODULE_B_COLORS.degrading,
    },
  ];


  /* =======================================================
     FINAL SUMMARY
  ======================================================= */

  const totalComponents =
    Number(
      results?.summary
        ?.total_components
    ) || components.length;

  const lowRisk =
    Number(
      results?.summary?.low_risk
    ) || 0;

  const mediumRisk =
    Number(
      results?.summary
        ?.medium_risk
    ) || 0;

  const highRisk =
    Number(
      results?.summary?.high_risk
    ) || 0;


  /* =======================================================
     BATCH INTELLIGENCE
  ======================================================= */

  const batchStats =
    useMemo(() => {

      const lookup = {};

      components.forEach(
        (component) => {

          const batchId =
            String(
              component.Batch_ID ||
                "Unknown"
            ).trim();

          if (!lookup[batchId]) {
            lookup[batchId] = {
              batchId,
              total: 0,
              low: 0,
              medium: 0,
              high: 0,
              anomaly: 0,
              warning: 0,
              normal: 0,
              earlyRejection: 0,
              initialReadings: [],
            };
          }

          const batch =
            lookup[batchId];

          batch.total += 1;

          const risk =
            getRiskInformation(
              component.Final_Risk
            );

          batch[
            risk.className
          ] += 1;

          const anomaly =
            getAnomalyInformation(
              component.Module_A_Risk
            );

          if (
            anomaly.className ===
            "anomaly"
          ) {
            batch.anomaly += 1;
          } else if (
            anomaly.className ===
            "warning"
          ) {
            batch.warning += 1;
          } else {
            batch.normal += 1;
          }

          const degradation =
            getDegradationInformation(
              component
                .Module_B_Decision
            );

          if (
            degradation.className ===
            "degrading"
          ) {
            batch.earlyRejection += 1;
          }

          const initial =
            Number(
              component.Value_0h_uA
            );

          if (
            Number.isFinite(initial)
          ) {
            batch.initialReadings.push(
              initial
            );
          }
        }
      );

      return Object.values(lookup)
        .map((batch) => ({
          ...batch,

          median0h:
            median(
              batch.initialReadings
            ),

          highRate:
            batch.total
              ? (batch.high /
                  batch.total) *
                100
              : 0,

          mediumRate:
            batch.total
              ? (batch.medium /
                  batch.total) *
                100
              : 0,

          anomalyRate:
            batch.total
              ? (batch.anomaly /
                  batch.total) *
                100
              : 0,

          earlyRejectionRate:
            batch.total
              ? (batch
                  .earlyRejection /
                  batch.total) *
                100
              : 0,
        }))
        .sort((a, b) => {

          if (
            b.highRate !==
            a.highRate
          ) {
            return (
              b.highRate -
              a.highRate
            );
          }

          return (
            b.anomalyRate -
            a.anomalyRate
          );
        });

    }, [components]);


  useEffect(() => {
    if (
      batchStats.length &&
      !selectedBatch
    ) {
      setSelectedBatch(
        batchStats[0].batchId
      );
    }
  }, [
    batchStats,
    selectedBatch,
  ]);


  const batchMedianLookup =
    useMemo(() => {

      const lookup = {};

      batchStats.forEach(
        (batch) => {
          lookup[batch.batchId] =
            batch.median0h;
        }
      );

      return lookup;

    }, [batchStats]);


  /* =======================================================
     SEARCH + FILTER
  ======================================================= */

  const filteredComponents =
    useMemo(() => {

      const query = search
        .trim()
        .toLowerCase();

      return components.filter(
        (component) => {

          const risk =
            getRiskInformation(
              component.Final_Risk
            );

          const matchesRisk =
            riskFilter === "all" ||
            risk.className ===
              riskFilter;

          if (!matchesRisk) {
            return false;
          }

          if (!query) {
            return true;
          }

          const searchable =
            [
              component.Component_ID,
              component.Batch_ID,
              risk.label,
              component.Final_Risk,
              component.Module_A_Risk,
              component
                .Module_B_Decision,
            ]
              .filter(Boolean)
              .join(" ")
              .toLowerCase();

          return searchable.includes(
            query
          );
        }
      );

    }, [
      components,
      search,
      riskFilter,
    ]);


  useEffect(() => {
    setCurrentPage(1);
    setPageJump("1");
  }, [
    search,
    riskFilter,
    rowsPerPage,
  ]);


  const totalPages =
    Math.max(
      1,
      Math.ceil(
        filteredComponents.length /
          rowsPerPage
      )
    );


  useEffect(() => {
    if (
      currentPage >
      totalPages
    ) {
      setCurrentPage(
        totalPages
      );

      setPageJump(
        String(totalPages)
      );
    }
  }, [
    currentPage,
    totalPages,
  ]);


  const pageStart =
    (currentPage - 1) *
    rowsPerPage;


  const visibleComponents =
    filteredComponents.slice(
      pageStart,
      pageStart + rowsPerPage
    );


  const visibleStart =
    filteredComponents.length
      ? pageStart + 1
      : 0;


  const visibleEnd =
    Math.min(
      pageStart + rowsPerPage,
      filteredComponents.length
    );


  const changePage = (page) => {

    const nextPage =
      Math.min(
        totalPages,
        Math.max(1, page)
      );

    setCurrentPage(
      nextPage
    );

    setPageJump(
      String(nextPage)
    );
  };


  const handlePageJump = () => {

    const requested =
      Number(pageJump);

    if (
      Number.isFinite(
        requested
      )
    ) {
      changePage(
        Math.round(requested)
      );
    }
  };


  const paginationNumbers =
    useMemo(() => {

      if (totalPages <= 7) {
        return Array.from(
          {
            length: totalPages,
          },
          (_, index) =>
            index + 1
        );
      }

      const pages = [1];

      if (currentPage > 4) {
        pages.push(
          "left-ellipsis"
        );
      }

      const start =
        Math.max(
          2,
          currentPage - 1
        );

      const end =
        Math.min(
          totalPages - 1,
          currentPage + 1
        );

      for (
        let page = start;
        page <= end;
        page += 1
      ) {
        pages.push(page);
      }

      if (
        currentPage <
        totalPages - 3
      ) {
        pages.push(
          "right-ellipsis"
        );
      }

      pages.push(
        totalPages
      );

      return pages;

    }, [
      totalPages,
      currentPage,
    ]);


  /* =======================================================
     SELECTED COMPONENT
  ======================================================= */

  const selectedComponentData =
    useMemo(() => {

      if (!selectedComponent) {
        return null;
      }

      return (
        components.find(
          (component) =>
            String(
              component.Component_ID
            ) ===
            String(
              selectedComponent
            )
        ) || null
      );

    }, [
      selectedComponent,
      components,
    ]);


  const selectedComponentBatchMedian =
    selectedComponentData
      ? batchMedianLookup[
          String(
            selectedComponentData.Batch_ID
          )
        ]
      : null;


  /* =======================================================
     RENDER
  ======================================================= */

  return (
    <div className="app">

      {/* HEADER */}

      <header className="header">

        <div>

          <div className="brand">

            <div className="logo">

              <div className="logo-glow" />

              <svg
                className="logo-svg"
                viewBox="0 0 40 40"
              >
                <circle
                  className="logo-ring"
                  cx="20"
                  cy="20"
                  r="17"
                />

                <path
                  className="logo-pulse"
                  d="M8 21H13L16 14L20 27L24 17L27 21H32"
                />
              </svg>

            </div>

            <span>
              OverWatch{" "}
              <strong>
                AI
              </strong>
            </span>

          </div>

          <p className="brand-subtitle">
            Intelligent Reliability
            Analysis
          </p>

        </div>


        <button
          type="button"
          className="theme-toggle"
          onClick={() =>
            setDarkMode(
              (previous) =>
                !previous
            )
          }
          aria-label="Toggle dark mode"
        >

          <span className="theme-toggle-track">

            <span className="theme-icon theme-icon-light">
              ☀
            </span>

            <span
              className={`theme-toggle-circle ${
                darkMode
                  ? "theme-toggle-circle-dark"
                  : ""
              }`}
            />

            <span className="theme-icon theme-icon-dark">
              ☾
            </span>

          </span>

        </button>

      </header>


      <main className="main">

        {/* INTRO */}

        <section className="intro">

          <span className="eyebrow">
            AI-POWERED RELIABILITY
          </span>

          <h1>
            Intelligent Component
            Reliability Analysis
          </h1>

          <p>
            Upload component burn-in
            measurements to detect
            unusual behavior, predict
            future degradation and
            understand which components
            need attention.
          </p>

        </section>


        {/* UPLOAD */}

        <section className="upload-card">

          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,text/csv"
            hidden
            onChange={(event) =>
              handleFile(
                event.target
                  .files?.[0]
              )
            }
          />


          <div
            className="drop-zone"
            onClick={() =>
              !loading &&
              fileInputRef.current?.click()
            }
            onDragOver={(event) =>
              event.preventDefault()
            }
            onDrop={(event) => {
              event.preventDefault();

              handleFile(
                event.dataTransfer
                  .files?.[0]
              );
            }}
          >

            <div className="upload-icon">
              <svg
                viewBox="0 0 48 48"
                aria-hidden="true"
              >
                <path
                  d="M13 5H29L38 14V40C38 41.7 36.7 43 35 43H13C11.3 43 10 41.7 10 40V8C10 6.3 11.3 5 13 5Z"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinejoin="round"
                />

                <path
                  d="M29 5V14H38"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinejoin="round"
                />

                <path
                  d="M24 34V20"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                />

                <path
                  d="M18.5 25.5L24 20L29.5 25.5"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>

            <h2>
              Select a CSV file to
              begin analysis
            </h2>

            <p>
              Drag and drop your
              component dataset here
            </p>

            <span className="browse">
              Browse files
            </span>


            {file && (

              <div
                className="file-info"
                onClick={(event) =>
                  event.stopPropagation()
                }
              >

                <div>
                  <strong>
                    {file.name}
                  </strong>

                  <span>
                    {(
                      file.size /
                      1024
                    ).toFixed(1)}{" "}
                    KB
                  </span>
                </div>

                <button
                  type="button"
                  className="remove-button"
                  onClick={
                    removeFile
                  }
                >
                  ×
                </button>

              </div>

            )}

          </div>


          <button
            type="button"
            className="execute-button"
            disabled={
              !file ||
              loading
            }
            onClick={
              handleAnalysis
            }
          >
            {loading
              ? "Analyzing components..."
              : "Execute Analysis"}
          </button>


          {error && (
            <p className="analysis-error">
              {error}
            </p>
          )}

        </section>


        {/* RESULTS */}

        {results?.summary && (

          <section
            className={`results-section ${
              showResultsAnimation
                ? "results-visible"
                : ""
            }`}
          >

            <div className="results-heading">

              <span className="results-eyebrow">
                ANALYSIS COMPLETE
              </span>

              <h2>
                Reliability Overview
              </h2>

              <p>
                A clear overview of
                unusual component
                behavior, predicted
                degradation and final
                reliability risk.
              </p>

            </div>


            {/* MODULE A + MODULE B */}

            <div className="module-analysis-grid">

              <article className="module-analysis-card module-a-card">

                <div className="module-card-heading">

                  <div className="module-heading-icon module-a-heading-icon">
                    ∿
                  </div>

                  <div>

                    <span className="module-card-eyebrow">
                      MODULE-A
                    </span>

                    <h3>
                      Anomaly Detection
                    </h3>

                    <p>
                      Finds components
                      that behave
                      differently from
                      others in the same
                      batch.
                    </p>

                  </div>

                </div>


                <div className="module-card-body">

                  <div className="module-donut">

                    <ResponsiveContainer
                      width="100%"
                      height="100%"
                    >
                      <PieChart>

                        <Pie
                          data={
                            moduleAData
                          }
                          dataKey="value"
                          innerRadius={60}
                          outerRadius={82}
                          paddingAngle={2}
                          stroke="none"
                          isAnimationActive
                        >
                          {moduleAData.map(
                            (entry) => (
                              <Cell
                                key={
                                  entry.name
                                }
                                fill={
                                  entry.color
                                }
                                stroke="none"
                              />
                            )
                          )}
                        </Pie>

                      </PieChart>
                    </ResponsiveContainer>

                    <div className="module-donut-center">
                      <strong>
                        {moduleATotal.toLocaleString()}
                      </strong>

                      <span>
                        Components
                      </span>
                    </div>

                  </div>


                  <div className="module-result-list">

                    {moduleAData.map(
                      (entry) => (

                        <div
                          className="module-result-row"
                          key={
                            entry.name
                          }
                        >

                          <div className="module-result-name">

                            <span
                              className="module-result-dot"
                              style={{
                                background:
                                  entry.color,
                              }}
                            />

                            <span>
                              {entry.name}
                            </span>

                          </div>

                          <div className="module-result-values">

                            <strong>
                              {entry.value.toLocaleString()}
                            </strong>

                            <span>
                              {getPercentage(
                                entry.value,
                                moduleATotal
                              )}
                              %
                            </span>

                          </div>

                        </div>

                      )
                    )}

                  </div>

                </div>

              </article>


              <article className="module-analysis-card module-b-card">

                <div className="module-card-heading">

                  <div className="module-heading-icon module-b-heading-icon">
                    ↗
                  </div>

                  <div>

                    <span className="module-card-eyebrow">
                      MODULE-B
                    </span>

                    <h3>
                      Degradation Prediction
                    </h3>

                    <p>
                      Uses early test
                      readings to predict
                      how the component
                      may behave by 168
                      hours.
                    </p>

                  </div>

                </div>


                <div className="module-card-body">

                  <div className="module-donut">

                    <ResponsiveContainer
                      width="100%"
                      height="100%"
                    >
                      <PieChart>

                        <Pie
                          data={
                            moduleBData
                          }
                          dataKey="value"
                          innerRadius={60}
                          outerRadius={82}
                          paddingAngle={2}
                          stroke="none"
                          isAnimationActive
                        >
                          {moduleBData.map(
                            (entry) => (
                              <Cell
                                key={
                                  entry.name
                                }
                                fill={
                                  entry.color
                                }
                                stroke="none"
                              />
                            )
                          )}
                        </Pie>

                      </PieChart>
                    </ResponsiveContainer>

                    <div className="module-donut-center">
                      <strong>
                        {moduleBTotal.toLocaleString()}
                      </strong>

                      <span>
                        Components
                      </span>
                    </div>

                  </div>


                  <div className="module-result-list">

                    {moduleBData.map(
                      (entry) => (

                        <div
                          className="module-result-row"
                          key={
                            entry.name
                          }
                        >

                          <div className="module-result-name">

                            <span
                              className="module-result-dot"
                              style={{
                                background:
                                  entry.color,
                              }}
                            />

                            <span>
                              {entry.name}
                            </span>

                          </div>

                          <div className="module-result-values">

                            <strong>
                              {entry.value.toLocaleString()}
                            </strong>

                            <span>
                              {getPercentage(
                                entry.value,
                                moduleBTotal
                              )}
                              %
                            </span>

                          </div>

                        </div>

                      )
                    )}

                  </div>

                </div>

              </article>

            </div>


            {/* FINAL RISK */}

            <div className="final-risk-heading">

              <span className="dashboard-eyebrow">
                FINAL RELIABILITY RESULT
              </span>

              <h2>
                What Needs Attention?
              </h2>

              <p>
                OverWatch combines the
                anomaly check and future
                degradation prediction
                into one simple
                reliability decision.
              </p>

            </div>


            <div className="summary-grid">

              <article className="summary-card summary-total">

                <span className="summary-card-icon" aria-hidden="true">
                  <svg viewBox="0 0 24 24">
                    <rect x="4" y="4" width="16" height="16" rx="3" />
                    <path d="M8 9H16" />
                    <path d="M8 13H16" />
                    <path d="M8 17H13" />
                    <circle cx="17" cy="17" r="1.5" />
                  </svg>
                </span>

                <div>
                  <span className="summary-label">
                    Total Components
                  </span>

                  <strong className="summary-value">
                    <AnimatedNumber
                      value={
                        totalComponents
                      }
                      start={
                        showResultsAnimation
                      }
                    />
                  </strong>

                  <span className="summary-description">
                    Components analyzed
                  </span>
                </div>

              </article>


              <article className="summary-card summary-low">

                <span className="summary-card-icon" aria-hidden="true">
                  <svg viewBox="0 0 24 24">
                    <path d="M12 3L19 6V11C19 15.6 16.2 19.3 12 21C7.8 19.3 5 15.6 5 11V6L12 3Z" />
                    <path d="M8.5 12L10.8 14.3L15.8 9.3" />
                  </svg>
                </span>

                <div>
                  <span className="summary-label">
                    Low Risk
                  </span>

                  <strong className="summary-value">
                    <AnimatedNumber
                      value={lowRisk}
                      start={
                        showResultsAnimation
                      }
                    />
                  </strong>

                  <span className="summary-description">
                    Stable components
                  </span>
                </div>

              </article>


              <article className="summary-card summary-review">

                <span className="summary-card-icon" aria-hidden="true">
                  <svg viewBox="0 0 24 24">
                    <path d="M12 4L21 20H3L12 4Z" />
                    <path d="M12 9V14" />
                    <circle cx="12" cy="17" r="1" />
                  </svg>
                </span>

                <div>
                  <span className="summary-label">
                    Needs Review
                  </span>

                  <strong className="summary-value">
                    <AnimatedNumber
                      value={
                        mediumRisk
                      }
                      start={
                        showResultsAnimation
                      }
                    />
                  </strong>

                  <span className="summary-description">
                    Medium-risk
                    components
                  </span>
                </div>

              </article>


              <article className="summary-card summary-high">

                <span className="summary-card-icon" aria-hidden="true">
                  <svg viewBox="0 0 24 24">
                    <path d="M12 3L21 8V13C21 17.5 17.8 20.5 12 22C6.2 20.5 3 17.5 3 13V8L12 3Z" />
                    <path d="M12 8V14" />
                    <circle cx="12" cy="17" r="1" />
                  </svg>
                </span>

                <div>
                  <span className="summary-label">
                    High Risk
                  </span>

                  <strong className="summary-value">
                    <AnimatedNumber
                      value={
                        highRisk
                      }
                      start={
                        showResultsAnimation
                      }
                    />
                  </strong>

                  <span className="summary-description">
                    Critical components
                  </span>
                </div>

              </article>

            </div>


            {/* BATCH HEALTH INTELLIGENCE */}

            <BatchHealthIntelligence
              batches={batchStats}
              selectedBatch={
                selectedBatch
              }
              onSelectBatch={
                setSelectedBatch
              }
            />


            {/* COMPONENT RESULTS */}

            <div className="component-results-layout">

              <section className="component-results-card">

                <div className="component-results-header">

                  <div className="component-results-title">

                    <span className="results-file-icon">
                      ▤
                    </span>

                    <div>
                      <h3>
                        Component Results
                      </h3>

                      <p>
                        Select any
                        component to see
                        its readings,
                        reliability result
                        and OverWatch
                        explanation.
                      </p>
                    </div>

                  </div>


                  <div className="results-search">

                    <span className="search-icon">
                      ⌕
                    </span>

                    <input
                      value={search}
                      onChange={(
                        event
                      ) =>
                        setSearch(
                          event.target
                            .value
                        )
                      }
                      placeholder="Search Component ID, Batch ID or risk..."
                    />

                    {search && (
                      <button
                        type="button"
                        className="clear-search"
                        onClick={() =>
                          setSearch("")
                        }
                      >
                        ×
                      </button>
                    )}

                  </div>

                </div>


                <div className="risk-filter-bar">

                  <button
                    type="button"
                    className={`risk-filter-button filter-all ${
                      riskFilter ===
                      "all"
                        ? "active"
                        : ""
                    }`}
                    onClick={() =>
                      setRiskFilter(
                        "all"
                      )
                    }
                  >
                    All (
                    {totalComponents.toLocaleString()}
                    )
                  </button>


                  <button
                    type="button"
                    className={`risk-filter-button filter-high ${
                      riskFilter ===
                      "high"
                        ? "active"
                        : ""
                    }`}
                    onClick={() =>
                      setRiskFilter(
                        "high"
                      )
                    }
                  >
                    ! High Risk (
                    {highRisk.toLocaleString()}
                    )
                  </button>


                  <button
                    type="button"
                    className={`risk-filter-button filter-medium ${
                      riskFilter ===
                      "medium"
                        ? "active"
                        : ""
                    }`}
                    onClick={() =>
                      setRiskFilter(
                        "medium"
                      )
                    }
                  >
                    ▲ Medium Risk (
                    {mediumRisk.toLocaleString()}
                    )
                  </button>


                  <button
                    type="button"
                    className={`risk-filter-button filter-low ${
                      riskFilter ===
                      "low"
                        ? "active"
                        : ""
                    }`}
                    onClick={() =>
                      setRiskFilter(
                        "low"
                      )
                    }
                  >
                    ✓ Low Risk (
                    {lowRisk.toLocaleString()}
                    )
                  </button>

                </div>


                <div className="component-table-wrapper">

                  <table className="component-results-table">

                    <thead>
                      <tr>

                        <th>
                          Component ID
                        </th>

                        <th>
                          Batch ID
                        </th>

                        <th>
                          Initial Reading
                          <span>
                            (0h µA)
                          </span>
                        </th>

                        <th>
                          After 24h
                          <span>
                            (µA)
                          </span>
                        </th>

                        <th>
                          After 96h
                          <span>
                            (µA)
                          </span>
                        </th>

                        <th>
                          After 168h
                          <span>
                            Actual (µA)
                          </span>
                        </th>

                        <th>
                          Predicted Reading
                          <span>
                            (168h µA)
                          </span>
                        </th>

                        <th>
                          Anomaly Status
                        </th>

                        <th>
                          Degradation
                          Status
                        </th>

                        <th>
                          Final Risk
                        </th>

                      </tr>
                    </thead>


                    <tbody>

                      {visibleComponents.map(
                        (
                          component,
                          index
                        ) => {

                          const anomaly =
                            getAnomalyInformation(
                              component.Module_A_Risk
                            );

                          const degradation =
                            getDegradationInformation(
                              component.Module_B_Decision
                            );

                          const risk =
                            getRiskInformation(
                              component.Final_Risk
                            );

                          const isSelected =
                            String(
                              selectedComponent
                            ) ===
                            String(
                              component.Component_ID
                            );

                          return (

                            <tr
                              key={
                                component.Component_ID ??
                                index
                              }
                              className={`component-clickable-row ${
                                isSelected
                                  ? "component-row-selected"
                                  : ""
                              }`}
                              onClick={() =>
                                setSelectedComponent(
                                  component.Component_ID
                                )
                              }
                            >

                              <td className="component-id-cell">
                                {component.Component_ID ||
                                  "—"}
                              </td>

                              <td>
                                {component.Batch_ID ||
                                  "—"}
                              </td>

                              <td className="reading-cell">
                                {formatNumber(
                                  component.Value_0h_uA
                                )}
                              </td>

                              <td className="reading-cell">
                                {formatNumber(
                                  component.Value_24h_uA
                                )}
                              </td>

                              <td className="reading-cell">
                                {formatNumber(
                                  component.Value_96h_uA
                                )}
                              </td>

                              <td className="reading-cell">
                                {formatNumber(
                                  component.Value_168h_uA
                                )}
                              </td>

                              <td className="reading-cell predicted-table-reading">
                                {formatNumber(
                                  component.Predicted_168h
                                )}
                              </td>

                              <td>
                                <span
                                  className={`status-pill anomaly-${anomaly.className}`}
                                >
                                  {anomaly.label}
                                </span>
                              </td>

                              <td>
                                <span
                                  className={`status-pill degradation-${degradation.className}`}
                                >
                                  {degradation.label}
                                </span>
                              </td>

                              <td>
                                <span
                                  className={`final-risk-pill final-${risk.className}`}
                                >
                                  {risk.label}
                                </span>
                              </td>

                            </tr>

                          );
                        }
                      )}


                      {visibleComponents.length ===
                        0 && (

                        <tr>
                          <td
                            className="no-results"
                            colSpan="10"
                          >
                            No matching
                            components
                            found.
                          </td>
                        </tr>

                      )}

                    </tbody>

                  </table>

                </div>


                <div className="pagination-footer">

                  <div className="pagination-summary">

                    Showing{" "}

                    <strong>
                      {visibleStart}–
                      {visibleEnd}
                    </strong>

                    {" "}of{" "}

                    <strong>
                      {filteredComponents.length.toLocaleString()}
                    </strong>

                    {" "}components

                  </div>


                  <div className="pagination-center">

                    <button
                      type="button"
                      className="pagination-arrow"
                      disabled={
                        currentPage === 1
                      }
                      onClick={() =>
                        changePage(
                          currentPage - 1
                        )
                      }
                    >
                      ‹
                    </button>


                    <div className="pagination-pages">

                      {paginationNumbers.map(
                        (
                          page,
                          index
                        ) => {

                          if (
                            typeof page ===
                            "string"
                          ) {
                            return (
                              <span
                                className="pagination-ellipsis"
                                key={`${page}-${index}`}
                              >
                                …
                              </span>
                            );
                          }

                          return (
                            <button
                              type="button"
                              key={page}
                              className={`pagination-page ${
                                currentPage ===
                                page
                                  ? "active"
                                  : ""
                              }`}
                              onClick={() =>
                                changePage(
                                  page
                                )
                              }
                            >
                              {page}
                            </button>
                          );
                        }
                      )}

                    </div>


                    <button
                      type="button"
                      className="pagination-arrow"
                      disabled={
                        currentPage ===
                        totalPages
                      }
                      onClick={() =>
                        changePage(
                          currentPage + 1
                        )
                      }
                    >
                      ›
                    </button>

                  </div>


                  <div className="pagination-controls">

                    <div className="page-jump">

                      <span>
                        Go to:
                      </span>

                      <input
                        type="number"
                        min="1"
                        max={
                          totalPages
                        }
                        value={
                          pageJump
                        }
                        onChange={(
                          event
                        ) =>
                          setPageJump(
                            event.target
                              .value
                          )
                        }
                        onKeyDown={(
                          event
                        ) => {
                          if (
                            event.key ===
                            "Enter"
                          ) {
                            handlePageJump();
                          }
                        }}
                      />

                      <button
                        type="button"
                        onClick={
                          handlePageJump
                        }
                      >
                        Go
                      </button>

                    </div>


                    <div className="rows-per-page">

                      <span>
                        Rows per page:
                      </span>

                      <select
                        value={
                          rowsPerPage
                        }
                        onChange={(
                          event
                        ) =>
                          setRowsPerPage(
                            Number(
                              event.target
                                .value
                            )
                          )
                        }
                      >
                        <option value={5}>
                          5
                        </option>

                        <option value={10}>
                          10
                        </option>

                        <option value={25}>
                          25
                        </option>

                        <option value={50}>
                          50
                        </option>
                      </select>

                    </div>

                  </div>

                </div>

                            </section>

            </div>


            {/* COMPONENT DETAILS - OPENS BELOW COMPONENT RESULTS */}

            {selectedComponentData && (

              <div className="component-details-below-wrapper">

                <ComponentDetailsPanel
                  component={
                    selectedComponentData
                  }
                  batchMedian={
                    selectedComponentBatchMedian
                  }
                  onClose={() =>
                    setSelectedComponent(
                      null
                    )
                  }
                />

              </div>

            )}

          </section>

        )}

      </main>

    </div>
  );
}