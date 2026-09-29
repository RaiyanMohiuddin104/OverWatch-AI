package com.overwatchai.model;

public class AnalysisResult {

    private String status;
    private Object moduleA;
    private Object moduleB;

    public AnalysisResult() {
    }

    public AnalysisResult(String status, Object moduleA, Object moduleB) {
        this.status = status;
        this.moduleA = moduleA;
        this.moduleB = moduleB;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public Object getModuleA() {
        return moduleA;
    }

    public void setModuleA(Object moduleA) {
        this.moduleA = moduleA;
    }

    public Object getModuleB() {
        return moduleB;
    }

    public void setModuleB(Object moduleB) {
        this.moduleB = moduleB;
    }
}