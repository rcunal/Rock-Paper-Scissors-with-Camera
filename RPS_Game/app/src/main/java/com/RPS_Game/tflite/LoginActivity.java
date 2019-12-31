package com.RPS_Game.tflite;

import android.content.Intent;
import android.os.Bundle;
import android.support.design.widget.Snackbar;
import android.support.design.widget.TextInputEditText;
import android.support.design.widget.TextInputLayout;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.AppCompatButton;
import android.support.v7.widget.AppCompatTextView;
import android.view.Gravity;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;


import com.RPS_Game.helpers.InputValidation;
import com.RPS_Game.sql.DatabaseHelper;

public class LoginActivity extends AppCompatActivity implements View.OnClickListener {
    private final AppCompatActivity activity = LoginActivity.this;


    private TextInputLayout textInputLayoutName;
    private TextInputLayout textInputLayoutPassword;

    private TextInputEditText textInputEditTextName;
    private TextInputEditText textInputEditTextPassword;

    private AppCompatButton appCompatButtonLogin;

    private AppCompatTextView textViewLinkRegister;

    private InputValidation inputValidation;
    private DatabaseHelper databaseHelper;

    private ImageView id;

    private Snackbar snackbar;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_login);
        getSupportActionBar().hide();

        initViews();
        initListeners();
        initObjects();


    }


    private void initViews() {

        textInputLayoutName = findViewById(R.id.textInputLayoutEmail);
        textInputLayoutPassword = findViewById(R.id.textInputLayoutPassword);

        textInputEditTextName = findViewById(R.id.textInputEditTextName);
        textInputEditTextPassword = findViewById(R.id.resultText);

        appCompatButtonLogin = findViewById(R.id.setScoreButton);

        textViewLinkRegister = findViewById(R.id.textViewLinkRegister);

        id = findViewById(R.id.icon);
    }


    private void initListeners() {
        appCompatButtonLogin.setOnClickListener(this);
        textViewLinkRegister.setOnClickListener(this);
    }


    private void initObjects() {
        databaseHelper = new DatabaseHelper(activity);
        inputValidation = new InputValidation(activity);

    }


    @Override
    public void onClick(View v) {
        switch (v.getId()) {
            case R.id.setScoreButton:
                verifyFromSQLite();
                break;
            case R.id.textViewLinkRegister:
                Intent intentRegister2 = new Intent(LoginActivity.this, RegisterActivity.class);
                startActivity(intentRegister2);
                break;
        }
    }


    private void verifyFromSQLite() {
        if (!inputValidation.isInputEditTextFilled(textInputEditTextName, textInputLayoutName, getString(R.string.error_message_email))) {
            return;
        }

        if (!inputValidation.isInputEditTextFilled(textInputEditTextPassword, textInputLayoutPassword, getString(R.string.error_message_email))) {
            return;
        }

        if (databaseHelper.checkUser(textInputEditTextName.getText().toString().trim()
                , textInputEditTextPassword.getText().toString().trim())) {
            Intent intentRegister1 = new Intent(LoginActivity.this, Lobi.class);
            intentRegister1.putExtra("NAME", textInputEditTextName.getText().toString().trim());
            emptyInputEditText();
            startActivity(intentRegister1);

        } else {
            Toast toast = Toast.makeText(getApplicationContext(),
                    R.string.error_valid_email_password, Toast.LENGTH_SHORT);
            toast.setGravity(Gravity.TOP | Gravity.CENTER_HORIZONTAL, 0, 0);
            toast.show();
        }
    }

    /**
     * This method is to empty all input edit text
     */
    private void emptyInputEditText() {
        textInputEditTextName.setText(null);
        textInputEditTextPassword.setText(null);
    }
}
