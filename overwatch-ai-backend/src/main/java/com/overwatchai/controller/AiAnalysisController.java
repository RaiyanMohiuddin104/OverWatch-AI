package com.overwatchai.controller;

import com.overwatchai.service.AiAnalysisService;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

@CrossOrigin(origins = {"http://localhost:5173", "http://localhost:5174"})
@RestController
@RequestMapping("/api/ai")
public class AiAnalysisController {

    private final AiAnalysisService aiAnalysisService;

    public AiAnalysisController(
            AiAnalysisService aiAnalysisService) {

        this.aiAnalysisService = aiAnalysisService;
    }


    // TEST MODULE A

    @GetMapping("/module-a")
    public String testModuleA() {

        return aiAnalysisService.testModuleA();
    }


    // TEST MODULE B

    @GetMapping("/module-b")
    public String testModuleB() {

        return aiAnalysisService.testModuleB();
    }


    // MODULE A ANALYSIS

    @PostMapping("/module-a/analyze")
    public String analyzeWithModuleA(
            @RequestParam("file") MultipartFile file) {

        return aiAnalysisService.analyzeWithModuleA(file);
    }


    // MODULE B PREDICTION

    @PostMapping("/module-b/predict")
    public String predictWithModuleB(
            @RequestParam("file") MultipartFile file) {

        return aiAnalysisService.predictWithModuleB(file);
    }


    // RUN BOTH MODULES

    @PostMapping("/both")
    public String runBothModules(
            @RequestParam("file") MultipartFile file) {

        return aiAnalysisService.runBothModules(file);
    }


    // TEST RISK ENGINE

    @GetMapping("/risk-test")
    public String testRiskEngine() {

        return aiAnalysisService.testRiskEngine();
    }


    // TEST MAP CONVERSION

    @PostMapping("/map-test")
    public Map<String, Object> testModuleAMap(
            @RequestParam("file") MultipartFile file) {

        return aiAnalysisService.testModuleAMap(file);
    }


    // --------------------------------------------------
    // COMPLETE OVERWATCH AI ANALYSIS
    // --------------------------------------------------

    @PostMapping("/risk-analysis")
    public Map<String, Object> runRiskAnalysis(
            @RequestParam("file") MultipartFile file) {

        return aiAnalysisService.runRiskAnalysis(file);
    }
}