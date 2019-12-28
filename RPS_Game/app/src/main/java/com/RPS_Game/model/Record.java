package com.RPS_Game.model;

public class Record {

    private String name;
    private String move;
    private String rmove;
    private String result;
    private String value;

    public Record(String name, String move, String rmove, String result,String value) {
        this.name = name;
        this.move = move;
        this.rmove = rmove;
        this.result = result;
        this.value = value;
    }


    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getMove() {
        return move;
    }

    public void setMove(String move) {
        this.move = move;
    }

    public String getRmove() {
        return rmove;
    }

    public void setRmove(String rmove) {
        this.rmove = rmove;
    }

    public String getResult() {
        return result;
    }

    public void setResult(String result) {
        this.result = result;
    }

    public String getValue() {
        return value;
    }

    public void setValue(String value) {
        this.value = value;
    }
}
