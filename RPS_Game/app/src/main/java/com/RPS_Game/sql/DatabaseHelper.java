package com.RPS_Game.sql;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.util.Log;

import com.RPS_Game.model.Record;
import com.RPS_Game.model.User;

import java.util.ArrayList;
import java.util.List;


public class DatabaseHelper extends SQLiteOpenHelper {

    // Database Version
    private static final int DATABASE_VERSION = 1;

    // Database Name
    private static final String DATABASE_NAME = "UserManager.db";

    // User table name
    private static final String TABLE_USER = "user";

    // User Table Columns names
    private static final String COLUMN_USER_ID = "user_id";
    private static final String COLUMN_USER_NAME = "user_name";
    private static final String COLUMN_USER_EMAIL = "user_email";
    private static final String COLUMN_USER_PASSWORD = "user_password";

    // create table sql query
    private String CREATE_USER_TABLE = "CREATE TABLE IF NOT EXISTS " + TABLE_USER + "("
            + COLUMN_USER_ID + " INTEGER ," + COLUMN_USER_NAME + " TEXT PRIMARY KEY,"
            + COLUMN_USER_EMAIL + " TEXT," + COLUMN_USER_PASSWORD + " TEXT" + ")";

    // drop table sql query
    private String DROP_USER_TABLE = "DROP TABLE IF EXISTS " + TABLE_USER;





    private static final String TABLE_RECORD = "record";

    // User Table Columns names

    private static final String COLUMN_USER_NAME1 = "user_name";
    private static final String COLUMN_USER_MOVE = "user_move";
    private static final String COLUMN_USER_RMOVE = "user_rmove";
    private static final String COLUMN_USER_RESULT = "user_result";
    private static final String COLUMN_USER_VALUE = "user_value";

    // create table sql query
    private String CREATE_RECORD_TABLE = "CREATE TABLE IF NOT EXISTS " + TABLE_RECORD + "("
            + COLUMN_USER_NAME1 + " TEXT,"
            + COLUMN_USER_MOVE + " TEXT," + COLUMN_USER_RMOVE + " TEXT," + COLUMN_USER_RESULT + " TEXT," + COLUMN_USER_VALUE + " TEXT" + ")";

    // drop table sql query
    private String DROP_RECORD_TABLE = "DROP TABLE IF EXISTS " + TABLE_RECORD;


    public DatabaseHelper(Context context) {
        super(context, DATABASE_NAME, null, DATABASE_VERSION);
        onCreate(this.getWritableDatabase());
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        db.execSQL(CREATE_USER_TABLE);
        db.execSQL(CREATE_RECORD_TABLE);
        //addnewTable();
    }

    public void addnewTable(){
        //At first you will need a Database object.Lets create it.
        SQLiteDatabase ourDatabase=this.getWritableDatabase();

        ourDatabase.execSQL(CREATE_RECORD_TABLE);//CreateTableString is the SQL Command String
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {

        //Drop User Table if exist
        db.execSQL(DROP_USER_TABLE);
        db.execSQL(DROP_RECORD_TABLE);
        // Create tables again
        onCreate(db);

    }


    public void addUser(User user) {
        SQLiteDatabase db = this.getWritableDatabase();

        ContentValues values = new ContentValues();
        values.put(COLUMN_USER_NAME, user.getName());
        values.put(COLUMN_USER_EMAIL, user.getEmail());
        values.put(COLUMN_USER_PASSWORD, user.getPassword());

        // Inserting Row
        db.insert(TABLE_USER, null, values);
        db.close();
    }


    public void addRecord(Record record) {
        SQLiteDatabase db = this.getWritableDatabase();
        System.out.println("İnserting to record table.");
        ContentValues values = new ContentValues();
        values.put(COLUMN_USER_NAME1, record.getName());
        values.put(COLUMN_USER_MOVE, record.getMove());
        values.put(COLUMN_USER_RMOVE, record.getRmove());
        values.put(COLUMN_USER_RESULT, record.getResult());
        values.put(COLUMN_USER_VALUE, record.getValue());

        // Inserting Row
        db.insert(TABLE_RECORD, null, values);
        db.close();
    }




    public List<User> getAllUser() {
        // array of columns to fetch
        String[] columns = {
                COLUMN_USER_ID,
                COLUMN_USER_EMAIL,
                COLUMN_USER_NAME,
                COLUMN_USER_PASSWORD
        };
        // sorting orders
        String sortOrder =
                COLUMN_USER_NAME + " ASC";
        List<User> userList = new ArrayList<User>();

        SQLiteDatabase db = this.getReadableDatabase();

        // query the user table
        Cursor cursor = db.query(TABLE_USER, //Table to query
                columns,    //columns to return
                null,        //columns for the WHERE clause
                null,        //The values for the WHERE clause
                null,       //group the rows
                null,       //filter by row groups
                sortOrder); //The sort order


        // Traversing through all rows and adding to list
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

        // return user list
        return userList;
    }

    public List<Record> getAllRecord(String name) {
        System.out.println("in getAllRecord");
        // array of columns to fetch
        String[] columns = {
                COLUMN_USER_NAME1,
                COLUMN_USER_MOVE,
                COLUMN_USER_RMOVE,
                COLUMN_USER_RESULT,
                COLUMN_USER_VALUE
        };
        // sorting orders

        String selection = COLUMN_USER_NAME1 + " = ?";
        String[] selectionArgs = {name};

        List<Record> recordList = new ArrayList<>();

        SQLiteDatabase db = this.getReadableDatabase();

        //query the user table
        Cursor cursor = db.query(TABLE_RECORD, //Table to query
                columns,    //columns to return
                selection,        //columns for the WHERE clause
                selectionArgs,        //The values for the WHERE clause
                null,       //group the rows
                null,       //filter by row groups
                null);
                 //The sort order

        //Cursor cursor = db.rawQuery("SELECT * FROM record WHERE COLUMN_USER_NAME1 = ?", new String[] {name});

        // Traversing through all rows and adding to list
        if (cursor.moveToFirst()) {
            do {
                Record record = new Record(COLUMN_USER_NAME1,COLUMN_USER_MOVE,COLUMN_USER_RMOVE,COLUMN_USER_RESULT,COLUMN_USER_VALUE);
//                user.setId(Integer.parseInt(cursor.getString(cursor.getColumnIndex(COLUMN_USER_ID))));
//                user.setName(cursor.getString(cursor.getColumnIndex(COLUMN_USER_NAME)));
//                user.setEmail(cursor.getString(cursor.getColumnIndex(COLUMN_USER_EMAIL)));
//                user.setPassword(cursor.getString(cursor.getColumnIndex(COLUMN_USER_PASSWORD)));
                record.setName(cursor.getString(cursor.getColumnIndex(COLUMN_USER_NAME1)));
                record.setMove(cursor.getString(cursor.getColumnIndex(COLUMN_USER_MOVE)));
                record.setRmove(cursor.getString(cursor.getColumnIndex(COLUMN_USER_RMOVE)));
                record.setResult(cursor.getString(cursor.getColumnIndex(COLUMN_USER_RESULT)));
                record.setValue(cursor.getString(cursor.getColumnIndex(COLUMN_USER_VALUE)));
                // Adding user record to list
                recordList.add(record);
            } while (cursor.moveToNext());
        }
        cursor.close();
        db.close();

        // return user list
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

        // updating row
        db.update(TABLE_USER, values, COLUMN_USER_ID + " = ?",
                new String[]{String.valueOf(user.getId())});
        db.close();
    }


    public void deleteUser(User user) {
        SQLiteDatabase db = this.getWritableDatabase();
        // delete user record by id
        db.delete(TABLE_USER, COLUMN_USER_ID + " = ?",
                new String[]{String.valueOf(user.getId())});
        db.close();
    }


    public boolean checkUser(String name) {

        // array of columns to fetch
        String[] columns = {
                COLUMN_USER_NAME
        };
        SQLiteDatabase db = this.getReadableDatabase();

        // selection criteria
        String selection = COLUMN_USER_NAME + " = ?";

        // selection argument
        String[] selectionArgs = {name};

        // query user table with condition

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

        // array of columns to fetch
        String[] columns = {
                COLUMN_USER_ID
        };
        SQLiteDatabase db = this.getReadableDatabase();
        // selection criteria
        String selection = COLUMN_USER_NAME + " = ?" + " AND " + COLUMN_USER_PASSWORD + " = ?";

        // selection arguments
        String[] selectionArgs = {name, password};

        // query user table with conditions

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