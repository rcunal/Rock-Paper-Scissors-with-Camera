package com.RPS_Game.sql;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

import com.RPS_Game.model.Record;
import com.RPS_Game.model.User;

import java.util.ArrayList;
import java.util.List;


public class DatabaseHelper extends SQLiteOpenHelper {

    private static final int DATABASE_VERSION = 1;

    private static final String DATABASE_NAME = "UserManager.db";

    private static final String TABLE_USER = "user";

    private static final String COLUMN_USER_ID = "user_id";
    private static final String COLUMN_USER_NAME = "user_name";
    private static final String COLUMN_USER_EMAIL = "user_email";
    private static final String COLUMN_USER_PASSWORD = "user_password";

    private String CREATE_USER_TABLE = "CREATE TABLE IF NOT EXISTS " + TABLE_USER + "("
            + COLUMN_USER_ID + " INTEGER ," + COLUMN_USER_NAME + " TEXT PRIMARY KEY,"
            + COLUMN_USER_EMAIL + " TEXT," + COLUMN_USER_PASSWORD + " TEXT" + ")";

    private String DROP_USER_TABLE = "DROP TABLE IF EXISTS " + TABLE_USER;





    private static final String TABLE_RECORD = "record";


    private static final String COLUMN_USER_NAME1 = "user_name";
    private static final String COLUMN_USER_MOVE = "user_move";
    private static final String COLUMN_USER_RMOVE = "user_rmove";
    private static final String COLUMN_USER_RESULT = "user_result";
    private static final String COLUMN_USER_VALUE = "user_value";

    private String CREATE_RECORD_TABLE = "CREATE TABLE IF NOT EXISTS " + TABLE_RECORD + "("
            + COLUMN_USER_NAME1 + " TEXT,"
            + COLUMN_USER_MOVE + " TEXT," + COLUMN_USER_RMOVE + " TEXT," + COLUMN_USER_RESULT + " TEXT," + COLUMN_USER_VALUE + " TEXT" + ")";

    private String DROP_RECORD_TABLE = "DROP TABLE IF EXISTS " + TABLE_RECORD;


    public DatabaseHelper(Context context) {
        super(context, DATABASE_NAME, null, DATABASE_VERSION);
        onCreate(this.getWritableDatabase());
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        db.execSQL(CREATE_USER_TABLE);
        db.execSQL(CREATE_RECORD_TABLE);

    }

    public void addnewTable(){
        SQLiteDatabase ourDatabase=this.getWritableDatabase();

        ourDatabase.execSQL(CREATE_RECORD_TABLE);//CreateTableString is the SQL Command String
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {

        db.execSQL(DROP_USER_TABLE);
        db.execSQL(DROP_RECORD_TABLE);
        onCreate(db);

    }


    public void addUser(User user) {
        SQLiteDatabase db = this.getWritableDatabase();

        ContentValues values = new ContentValues();
        values.put(COLUMN_USER_NAME, user.getName());
        values.put(COLUMN_USER_EMAIL, user.getEmail());
        values.put(COLUMN_USER_PASSWORD, user.getPassword());

        db.insert(TABLE_USER, null, values);
        db.close();
    }


    public void addRecord(Record record) {
        SQLiteDatabase db = this.getWritableDatabase();
        ContentValues values = new ContentValues();
        values.put(COLUMN_USER_NAME1, record.getName());
        values.put(COLUMN_USER_MOVE, record.getMove());
        values.put(COLUMN_USER_RMOVE, record.getRmove());
        values.put(COLUMN_USER_RESULT, record.getResult());
        values.put(COLUMN_USER_VALUE, record.getValue());

        db.insert(TABLE_RECORD, null, values);
        db.close();
    }




    public List<User> getAllUser() {
        String[] columns = {
                COLUMN_USER_ID,
                COLUMN_USER_EMAIL,
                COLUMN_USER_NAME,
                COLUMN_USER_PASSWORD
        };
        String sortOrder =
                COLUMN_USER_NAME + " ASC";
        List<User> userList = new ArrayList<User>();

        SQLiteDatabase db = this.getReadableDatabase();

        Cursor cursor = db.query(TABLE_USER, //Table to query
                columns,    //columns to return
                null,        //columns for the WHERE clause
                null,        //The values for the WHERE clause
                null,       //group the rows
                null,       //filter by row groups
                sortOrder); //The sort order


        if (cursor.moveToFirst()) {
            do {
                User user = new User();
                user.setId(Integer.parseInt(cursor.getString(cursor.getColumnIndex(COLUMN_USER_ID))));
                user.setName(cursor.getString(cursor.getColumnIndex(COLUMN_USER_NAME)));
                user.setEmail(cursor.getString(cursor.getColumnIndex(COLUMN_USER_EMAIL)));
                user.setPassword(cursor.getString(cursor.getColumnIndex(COLUMN_USER_PASSWORD)));
                // Adding user record to list
                userList.add(user);
            } while (cursor.moveToNext());
        }
        cursor.close();
        db.close();

        return userList;
    }

    public List<Record> getAllRecord(String name) {
        String[] columns = {
                COLUMN_USER_NAME1,
                COLUMN_USER_MOVE,
                COLUMN_USER_RMOVE,
                COLUMN_USER_RESULT,
                COLUMN_USER_VALUE
        };

        String selection = COLUMN_USER_NAME1 + " = ?";
        String[] selectionArgs = {name};

        List<Record> recordList = new ArrayList<>();

        SQLiteDatabase db = this.getReadableDatabase();

        Cursor cursor = db.query(TABLE_RECORD, //Table to query
                columns,    //columns to return
                selection,        //columns for the WHERE clause
                selectionArgs,        //The values for the WHERE clause
                null,       //group the rows
                null,       //filter by row groups
                null);
                 //The sort order

        if (cursor.moveToFirst()) {
            do {
                Record record = new Record(COLUMN_USER_NAME1,COLUMN_USER_MOVE,COLUMN_USER_RMOVE,COLUMN_USER_RESULT,COLUMN_USER_VALUE);
                record.setName(cursor.getString(cursor.getColumnIndex(COLUMN_USER_NAME1)));
                record.setMove(cursor.getString(cursor.getColumnIndex(COLUMN_USER_MOVE)));
                record.setRmove(cursor.getString(cursor.getColumnIndex(COLUMN_USER_RMOVE)));
                record.setResult(cursor.getString(cursor.getColumnIndex(COLUMN_USER_RESULT)));
                record.setValue(cursor.getString(cursor.getColumnIndex(COLUMN_USER_VALUE)));
                recordList.add(record);
            } while (cursor.moveToNext());
        }
        cursor.close();
        db.close();

        return recordList;
    }

    public String getScore(String name){

        String score = "";
        int index_SUM;

        SQLiteDatabase db = this.getReadableDatabase();
        String[] columns = {"SUM("+COLUMN_USER_VALUE+")"}; //{"SUM("+KEY_CONTENT2+")" };
        String selection = COLUMN_USER_NAME1 + " = ?";
        String[] selectionArgs = {name};

        Cursor cursor = db.query(TABLE_RECORD,
                columns,
                selection,
                selectionArgs,
                null,
                null,
                null);

        index_SUM = cursor.getColumnIndex("SUM("+COLUMN_USER_VALUE+")");
        for (cursor.moveToFirst(); !(cursor.isAfterLast()); cursor.moveToNext()) {
            score = score + cursor.getString(index_SUM);
        }
        return score;
    }

    public void resetScore(String name){
        SQLiteDatabase db = this.getReadableDatabase();
        db.delete(TABLE_RECORD,COLUMN_USER_NAME1 + "=?", new String[]{name});
    }

    public void updateUser(User user) {
        SQLiteDatabase db = this.getWritableDatabase();

        ContentValues values = new ContentValues();
        values.put(COLUMN_USER_NAME, user.getName());
        values.put(COLUMN_USER_EMAIL, user.getEmail());
        values.put(COLUMN_USER_PASSWORD, user.getPassword());

        db.update(TABLE_USER, values, COLUMN_USER_ID + " = ?",
                new String[]{String.valueOf(user.getId())});
        db.close();
    }


    public void deleteUser(User user) {
        SQLiteDatabase db = this.getWritableDatabase();
        db.delete(TABLE_USER, COLUMN_USER_ID + " = ?",
                new String[]{String.valueOf(user.getId())});
        db.close();
    }


    public boolean checkUser(String name) {

        String[] columns = {
                COLUMN_USER_NAME
        };
        SQLiteDatabase db = this.getReadableDatabase();

        String selection = COLUMN_USER_NAME + " = ?";

        String[] selectionArgs = {name};

        Cursor cursor = db.query(TABLE_USER, //Table to query
                columns,                    //columns to return
                selection,                  //columns for the WHERE clause
                selectionArgs,              //The values for the WHERE clause
                null,                       //group the rows
                null,                      //filter by row groups
                null);                      //The sort order
        int cursorCount = cursor.getCount();
        cursor.close();
        db.close();

        if (cursorCount > 0) {
            return true;
        }

        return false;
    }


    public boolean checkUser(String name, String password) {

        String[] columns = {
                COLUMN_USER_ID
        };
        SQLiteDatabase db = this.getReadableDatabase();
        String selection = COLUMN_USER_NAME + " = ?" + " AND " + COLUMN_USER_PASSWORD + " = ?";

        String[] selectionArgs = {name, password};

        Cursor cursor = db.query(TABLE_USER, //Table to query
                columns,                    //columns to return
                selection,                  //columns for the WHERE clause
                selectionArgs,              //The values for the WHERE clause
                null,                       //group the rows
                null,                       //filter by row groups
                null);                      //The sort order

        int cursorCount = cursor.getCount();

        cursor.close();
        db.close();
        if (cursorCount > 0) {
            return true;
        }

        return false;
    }
}