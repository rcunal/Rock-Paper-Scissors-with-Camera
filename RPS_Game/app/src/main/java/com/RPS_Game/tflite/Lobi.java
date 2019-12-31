package com.RPS_Game.tflite;

import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.AppCompatButton;
import android.view.View;
import android.widget.TextView;

import com.RPS_Game.sql.DatabaseHelper;

public class Lobi extends AppCompatActivity {

    DatabaseHelper databaseHelper;

    private AppCompatButton gameButton;
    private AppCompatButton scoreButton;
    private AppCompatButton setScoreButton;
    private TextView textViewName;
    private TextView scoreText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_lobi);

        databaseHelper = new DatabaseHelper(Lobi.this);

        gameButton = findViewById(R.id.gameButton);
        scoreButton = findViewById(R.id.scoreButton);
        textViewName = findViewById(R.id.randomMove);
        scoreText = findViewById(R.id.resultText);
        setScoreButton = findViewById(R.id.setScoreButton);

        final String nameFromIntent = getIntent().getStringExtra("NAME");
        textViewName.setText("Kullanıcı adı: "+nameFromIntent);

        setScoreButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
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
