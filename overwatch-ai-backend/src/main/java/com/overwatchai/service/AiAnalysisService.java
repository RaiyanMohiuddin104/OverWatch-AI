package com.overwatchai.service;

import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.util.*;

@Service
public class AiAnalysisService {

    private final RestTemplate restTemplate;
    private final RiskEngineService riskEngineService;

    public AiAnalysisService(RiskEngineService riskEngineService) {
        this.restTemplate = new RestTemplate();
        this.riskEngineService = riskEngineService;
    }


    // --------------------------------------------------
    // TEST MODULE A
    // --------------------------------------------------

    public String testModuleA() {
        String url = "http://localhost:8001/";
        return restTemplate.getForObject(url, String.class);
    }


    // --------------------------------------------------
    // TEST MODULE B
    // --------------------------------------------------

    public String testModuleB() {
        String url = "http://localhost:8000/";
        return restTemplate.getForObject(url, String.class);
    }


    // --------------------------------------------------
    // SEND CSV TO MODULE A
    // --------------------------------------------------

    public String analyzeWithModuleA(MultipartFile file) {

        String url = "http://localhost:8001/analyze";

        try {

            ByteArrayResource resource =
                    createFileResource(file);

            MultiValueMap<String, Object> body =
                    new LinkedMultiValueMap<>();

            body.add("file", resource);

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            HttpEntity<MultiValueMap<String, Object>> request =
                    new HttpEntity<>(body, headers);

            ResponseEntity<String> response =
                    restTemplate.postForEntity(
                            url,
                            request,
                            String.class
                    );

            return response.getBody();

        } catch (Exception e) {

            throw new RuntimeException(
                    "Error communicating with Module A: "
                            + e.getMessage()
            );
        }
    }


    // --------------------------------------------------
    // SEND CSV TO MODULE B
    // --------------------------------------------------

    public String predictWithModuleB(MultipartFile file) {

        String url = "http://localhost:8000/predict";

        try {

            ByteArrayResource resource =
                    createFileResource(file);

            MultiValueMap<String, Object> body =
                    new LinkedMultiValueMap<>();

            body.add("file", resource);

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            HttpEntity<MultiValueMap<String, Object>> request =
                    new HttpEntity<>(body, headers);

            ResponseEntity<String> response =
                    restTemplate.postForEntity(
                            url,
                            request,
                            String.class
                    );

            return response.getBody();

        } catch (Exception e) {

            throw new RuntimeException(
                    "Error communicating with Module B: "
                            + e.getMessage()
            );
        }
    }


    // --------------------------------------------------
    // RUN BOTH AI MODULES
    // --------------------------------------------------

    public String runBothModules(MultipartFile file) {

        String moduleAResult =
                analyzeWithModuleA(file);

        String moduleBResult =
                predictWithModuleB(file);

        return """
                {
                  "moduleA": %s,
                  "moduleB": %s
                }
                """.formatted(
                moduleAResult,
                moduleBResult
        );
    }


    // --------------------------------------------------
    // TEST RISK ENGINE
    // --------------------------------------------------

    public String testRiskEngine() {

        Map<String, Object> moduleA = Map.of(
                "Risk_Level", "Anomaly"
        );

        Map<String, Object> moduleB = Map.of(
                "Decision", "EARLY REJECTION"
        );

        return riskEngineService.calculateRisk(
                moduleA,
                moduleB
        );
    }


    // --------------------------------------------------
    // GET MODULE A RESPONSE AS MAP
    // --------------------------------------------------

    private Map<String, Object> getModuleAAsMap(
            MultipartFile file) {

        return sendFileAndGetMap(
                "http://localhost:8001/analyze",
                file
        );
    }


    // --------------------------------------------------
    // GET MODULE B RESPONSE AS MAP
    // --------------------------------------------------

    private Map<String, Object> getModuleBAsMap(
            MultipartFile file) {

        return sendFileAndGetMap(
                "http://localhost:8000/predict",
                file
        );
    }


    // --------------------------------------------------
    // TEST MAP CONVERSION
    // --------------------------------------------------

    public Map<String, Object> testModuleAMap(
            MultipartFile file) {

        return getModuleAAsMap(file);
    }


    // --------------------------------------------------
    // REAL RISK ANALYSIS
    // --------------------------------------------------

    public Map<String, Object> runRiskAnalysis(
            MultipartFile file) {

        // Run both AI models
        Map<String, Object> moduleAResponse =
                getModuleAAsMap(file);

        Map<String, Object> moduleBResponse =
                getModuleBAsMap(file);


        // Get results arrays
        Object moduleAResultsObject =
                moduleAResponse.get("results");

        Object moduleBResultsObject =
                moduleBResponse.get("results");


        if (!(moduleAResultsObject instanceof List<?>)
                || !(moduleBResultsObject instanceof List<?>)) {

            throw new RuntimeException(
                    "AI modules did not return valid results arrays."
            );
        }


        List<?> moduleAResults =
                (List<?>) moduleAResultsObject;

        List<?> moduleBResults =
                (List<?>) moduleBResultsObject;


        // --------------------------------------------------
        // INDEX MODULE B BY COMPONENT ID
        // --------------------------------------------------

        Map<String, Map<String, Object>> moduleBByComponent =
                new HashMap<>();

        for (Object object : moduleBResults) {

            if (object instanceof Map<?, ?> rawMap) {

                Map<String, Object> component =
                        convertToStringMap(rawMap);

                String componentId =
                        String.valueOf(
                                component.get("Component_ID")
                        );

                moduleBByComponent.put(
                        componentId,
                        component
                );
            }
        }


        // --------------------------------------------------
        // FINAL RESULTS
        // --------------------------------------------------

        List<Map<String, Object>> finalResults =
                new ArrayList<>();

        int lowRisk = 0;
        int mediumRisk = 0;
        int highRisk = 0;
        int unmatched = 0;


        // --------------------------------------------------
        // MATCH MODULE A WITH MODULE B
        // --------------------------------------------------

        for (Object object : moduleAResults) {

            if (!(object instanceof Map<?, ?> rawMap)) {
                continue;
            }

            Map<String, Object> moduleAComponent =
                    convertToStringMap(rawMap);

            String componentId =
                    String.valueOf(
                            moduleAComponent.get("Component_ID")
                    );

            Map<String, Object> moduleBComponent =
                    moduleBByComponent.get(componentId);


            if (moduleBComponent == null) {

                unmatched++;
                continue;
            }


            // Run Risk Engine
            String finalRisk =
                    riskEngineService.calculateRisk(
                            moduleAComponent,
                            moduleBComponent
                    );


            // Build final component response
            Map<String, Object> result =
                    new LinkedHashMap<>();

            result.put(
                    "Component_ID",
                    componentId
            );

            result.put(
                    "Batch_ID",
                    moduleAComponent.get("Batch_ID")
            );

            result.put(
                    "Value_0h_uA",
                    moduleAComponent.get("Value_0h_uA")
            );

            result.put(
                    "Module_A_Risk",
                    moduleAComponent.get("Risk_Level")
            );

            result.put(
                    "Module_A_Class",
                    moduleAComponent.get("Predicted_Class")
            );

            result.put(
                    "Dynamic_Anomaly_Score",
                    moduleAComponent.get(
                            "Dynamic_Anomaly_Score"
                    )
            );

            result.put(
                    "Module_B_Decision",
                    moduleBComponent.get("Decision")
            );

            result.put(
                    "Predicted_168h",
                    moduleBComponent.get("Predicted_168h")
            );

            result.put(
                    "Drift_Rate",
                    moduleBComponent.get("Drift_Rate")
            );

            result.put(
                    "Safety_Threshold",
                    moduleBComponent.get("Safety_Threshold")
            );

            result.put(
                    "Final_Risk",
                    finalRisk
            );

            finalResults.add(result);


            // Count risk levels
            switch (finalRisk) {

                case "HIGH RISK" ->
                        highRisk++;

                case "MEDIUM RISK" ->
                        mediumRisk++;

                case "LOW RISK" ->
                        lowRisk++;
            }
        }


        // --------------------------------------------------
        // CREATE SUMMARY
        // --------------------------------------------------

        Map<String, Object> summary =
                new LinkedHashMap<>();

        summary.put(
                "total_components",
                finalResults.size()
        );

        summary.put(
                "low_risk",
                lowRisk
        );

        summary.put(
                "medium_risk",
                mediumRisk
        );

        summary.put(
                "high_risk",
                highRisk
        );

        summary.put(
                "unmatched_components",
                unmatched
        );


        // --------------------------------------------------
        // FINAL API RESPONSE
        // --------------------------------------------------

        Map<String, Object> response =
                new LinkedHashMap<>();

        response.put(
                "status",
                "success"
        );

        response.put(
                "summary",
                summary
        );

        response.put(
                "results",
                finalResults
        );

        return response;
    }


    // --------------------------------------------------
    // HELPER: SEND FILE AND RECEIVE MAP
    // --------------------------------------------------

    private Map<String, Object> sendFileAndGetMap(
            String url,
            MultipartFile file) {

        try {

            ByteArrayResource resource =
                    createFileResource(file);

            MultiValueMap<String, Object> body =
                    new LinkedMultiValueMap<>();

            body.add("file", resource);

            HttpHeaders headers =
                    new HttpHeaders();

            headers.setContentType(
                    MediaType.MULTIPART_FORM_DATA
            );

            HttpEntity<MultiValueMap<String, Object>> request =
                    new HttpEntity<>(body, headers);

            ResponseEntity<Map> response =
                    restTemplate.postForEntity(
                            url,
                            request,
                            Map.class
                    );

            if (response.getBody() == null) {

                throw new RuntimeException(
                        "Empty response received from AI module."
                );
            }

            return response.getBody();

        } catch (Exception e) {

            throw new RuntimeException(
                    "Error communicating with AI module: "
                            + e.getMessage()
            );
        }
    }


    // --------------------------------------------------
    // HELPER: CREATE FILE RESOURCE
    // --------------------------------------------------

    private ByteArrayResource createFileResource(
            MultipartFile file) throws Exception {

        return new ByteArrayResource(
                file.getBytes()
        ) {

            @Override
            public String getFilename() {
                return file.getOriginalFilename();
            }
        };
    }


    // --------------------------------------------------
    // HELPER: CONVERT MAP
    // --------------------------------------------------

    private Map<String, Object> convertToStringMap(
            Map<?, ?> rawMap) {

        Map<String, Object> converted =
                new LinkedHashMap<>();

        for (Map.Entry<?, ?> entry :
                rawMap.entrySet()) {

            converted.put(
                    String.valueOf(entry.getKey()),
                    entry.getValue()
            );
        }

        return converted;
    }
}