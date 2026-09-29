package com.overwatchai.service;

import org.springframework.stereotype.Service;

import java.util.Map;

@Service
public class RiskEngineService {

    public String calculateRisk(
            Map<String, Object> moduleAResult,
            Map<String, Object> moduleBResult) {

        String moduleARisk =
                String.valueOf(moduleAResult.get("Risk_Level"));

        String moduleBDecision =
                String.valueOf(moduleBResult.get("Decision"));


        // --------------------------------------------------
        // HIGH RISK
        // --------------------------------------------------

        if ("Anomaly".equalsIgnoreCase(moduleARisk)
                && "EARLY REJECTION".equalsIgnoreCase(moduleBDecision)) {

            return "HIGH RISK";
        }


        // --------------------------------------------------
        // MEDIUM RISK
        // --------------------------------------------------

        if ("Anomaly".equalsIgnoreCase(moduleARisk)
                || "Warning".equalsIgnoreCase(moduleARisk)
                || "EARLY REJECTION".equalsIgnoreCase(moduleBDecision)) {

            return "MEDIUM RISK";
        }


        // --------------------------------------------------
        // LOW RISK
        // --------------------------------------------------

        return "LOW RISK";
    }
}