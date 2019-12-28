package com.RPS_Game.tflite;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;


import com.RPS_Game.model.Record;
import com.RPS_Game.sql.DatabaseHelper;

import java.util.Random;

public class Result extends AppCompatActivity {

    private Button goMain;
    private Button goLobi;
    private TextView textResult;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);

        goMain = findViewById(R.id.goMain);
        goLobi = findViewById(R.id.goLobi);
        textResult = findViewById(R.id.textResult);

        Intent intent = getIntent();
        String id = intent.getStringExtra("RESULT");
        final String nameFromIntent = getIntent().getStringExtra("NAME");

        goMain.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intentRegister1 = new Intent(Result.this, MainActivity.class);
                intentRegister1.putExtra("NAME", nameFromIntent);
                startActivity(intentRegister1);
            }
        });

        goLobi.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intentRegister1 = new Intent(Result.this, Lobi.class);
                intentRegister1.putExtra("NAME", nameFromIntent);
                startActivity(intentRegister1);
            }
        });



        System.out.println("id " + id); //+++++++++++++++++

        int result = -10;
        int[][] mtr;
        mtr = new int[][]{{0, -1, 1}, {1, 0, -1}, {-1, 1, 0}};
        int sonuc;

        id = id.split(" ")[0];
        System.out.println("id22 " + id); //+++++++++++++++++


        if( id.equals("rock") ){
            //0
            result = 0;
        }
        else if( id.equals("paper") ){
            //1
            result = 1;
        }
        else if( id.equals("scissors") ) {
            //2
            result = 2;
        }
        System.out.println("id11111 " + result);
        int random = new Random().nextInt(3);
        System.out.println("id11111 " + random);

        sonuc = mtr[result][random];

        String randomString = "";
        String resultString = "";
        if (random == 0){
            randomString = randomString + "rock";
        }
        else if(random == 1){
            randomString = randomString + "paper";
        }
        else if(random == 2){
            randomString = randomString + "scissors";
        }

        System.out.println("id11111 " + sonuc);


        if (sonuc == 0){
            resultString = resultString + "Berabere";
        }
        else if(sonuc == 1){
            resultString = resultString + "Kazandın";
        }
        else if(sonuc == -1){
            resultString = resultString + "Kaybettin";
        }
        textResult.setText(" " + " " + resultString);
        //String name = "ozcan";
        //Record record = new Record(nameFromIntent,String.valueOf(result),String.valueOf(random),String.valueOf(sonuc));
        Record record = new Record(nameFromIntent,id,randomString,resultString,String.valueOf(sonuc));
        DatabaseHelper databaseHelper = new DatabaseHelper(Result.this);
        databaseHelper.addRecord(record);

//                +++++++++++++++++++++++++++++++++++++++++++



    }
}
