package com.RPS_Game.tflite;

import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import com.RPS_Game.sql.DatabaseHelper;

public class Lobi extends AppCompatActivity {

    DatabaseHelper databaseHelper;

    private Button gameButton;
    private Button scoreButton;
    private Button setScoreButton;
    private TextView textViewName;
    private TextView scoreText;
    private String sumScore;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_lobi);

        databaseHelper = new DatabaseHelper(Lobi.this);

        gameButton = findViewById(R.id.gameButton);
        scoreButton = findViewById(R.id.scoreButton);
        textViewName = findViewById(R.id.textViewName);
        scoreText = findViewById(R.id.scoreText);
        setScoreButton = findViewById(R.id.setScoreButton);

        final String nameFromIntent = getIntent().getStringExtra("NAME");
        textViewName.setText("Kullanıcı adı: "+nameFromIntent);

        setScoreButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                /*sumScore = databaseHelper.getScore(nameFromIntent);
                if(sumScore.equals("")){

                    scoreText.setText("Henüz Skorunuz yok Oyuna Başlayın :)");
                }else{
                    scoreText.setText("Skorunuz :" + sumScore);
                }*/
                scoreText.setText("Skorunuz :" + databaseHelper.getScore(nameFromIntent));
            }
        });

        gameButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                Intent intentRegister1 = new Intent(Lobi.this, MainActivity.class);
                intentRegister1.putExtra("NAME", nameFromIntent);
                startActivity(intentRegister1);
            }
        });

        scoreButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                Intent intentRegister2 = new Intent(Lobi.this, RecordActivity.class);
                intentRegister2.putExtra("NAME", nameFromIntent);
                startActivity(intentRegister2);
            }
        });

    }
}
