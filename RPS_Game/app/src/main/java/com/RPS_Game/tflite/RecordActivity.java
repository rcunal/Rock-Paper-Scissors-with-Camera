package com.RPS_Game.tflite;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.os.AsyncTask;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.AppCompatTextView;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import com.RPS_Game.adapters.RecordAdapter;
import com.RPS_Game.model.Record;
import com.RPS_Game.model.User;
import com.RPS_Game.sql.DatabaseHelper;

import java.util.ArrayList;
import java.util.List;

public class RecordActivity extends AppCompatActivity {

    private AppCompatTextView textViewName;
    private RecyclerView recyclerViewUsers;
    private Button clearButton;
    private List<Record> listRecords;
    private DatabaseHelper databaseHelper;
    private RecordAdapter recordAdapter;
    private List<User> userList;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_record);
        getSupportActionBar().setTitle("");
        initViews();
        initObjects();
        final String nameFromIntent = getIntent().getStringExtra("NAME");

        clearButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                databaseHelper.resetScore(nameFromIntent);
                Toast toast = Toast.makeText(getApplicationContext(), "Gecmiş Sıfırlandı", Toast.LENGTH_LONG);
                toast.show();
                /*Handler handler = new Handler();
                handler.postDelayed(new Runnable() {
                    public void run() {
                        // yourMethod();
                    }
                }, 500);   //0.5 seconds*/
                Intent intentRegister1 = new Intent(RecordActivity.this, Lobi.class);
                intentRegister1.putExtra("NAME", nameFromIntent);
                startActivity(intentRegister1);
            }
        });
    }

    private void initViews() {
        textViewName = (AppCompatTextView) findViewById(R.id.randomMove);
        recyclerViewUsers = (RecyclerView) findViewById(R.id.recyclerViewUsers);
        clearButton = findViewById(R.id.clearButton);
    }

    private void initObjects() {
        listRecords = new ArrayList<>();

        userList = new ArrayList<>();

        databaseHelper = new DatabaseHelper(RecordActivity.this);


        recyclerViewUsers =  (RecyclerView) findViewById(R.id.recyclerViewUsers);

        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this);
        linearLayoutManager.setOrientation(LinearLayoutManager.VERTICAL);
        recyclerViewUsers.setLayoutManager(linearLayoutManager);

        recordAdapter = new RecordAdapter(listRecords);
        recyclerViewUsers.setAdapter(recordAdapter);
        recordAdapter.notifyDataSetChanged();

        getDataFromSQLite();


    }



    @SuppressLint("StaticFieldLeak")
    private void getDataFromSQLite() {

        System.out.println("in getDataFromSqlite");


        new AsyncTask<Void, Void, Void>() {

            @Override
            protected Void doInBackground(Void... params) {
                listRecords.clear();
                final String nameFromIntent = getIntent().getStringExtra("NAME");
                listRecords.addAll(databaseHelper.getAllRecord(nameFromIntent));
                System.out.println("after addAll");
                return null;
            }

            @Override
            protected void onPostExecute(Void aVoid) {
                super.onPostExecute(aVoid);
                recordAdapter.notifyDataSetChanged();
                System.out.println("after notifyDataChanged");
            }

            @Override
            protected void onProgressUpdate(Void... values) {
                super.onProgressUpdate(values);
            }
        }.execute();
    }
}
