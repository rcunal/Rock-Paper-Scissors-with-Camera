package com.RPS_Game.adapters;

import android.content.Context;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import com.RPS_Game.model.Record;
import com.RPS_Game.tflite.R;

import java.util.List;

/**
 * Created by qosmio on 15.01.2019.
 */

public class RecordAdapter extends RecyclerView.Adapter<RecordAdapter.MyViewHolder> {

    private List<Record> RecordList;
    private Context mContext;

    public class MyViewHolder extends RecyclerView.ViewHolder implements View.OnClickListener{

        public TextView textViewName, textViewMove, textViewRMove, textViewResult;


        public MyViewHolder(View view) {
            super(view);
            view.setOnClickListener(this);
            textViewName = (TextView) view.findViewById(R.id.textViewName);
            textViewMove = (TextView) view.findViewById(R.id.textViewMove);
            textViewRMove = (TextView) view.findViewById(R.id.textViewRmove);
            textViewResult = (TextView) view.findViewById(R.id.textViewResult1);
        }

        @Override
        public void onClick(View view) {

        }
    }


    public RecordAdapter(List<Record> recordList) {
        this.RecordList = recordList;
    }

    @Override
    public MyViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        View itemView = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_user_recycler, parent, false);
        mContext = parent.getContext();
        return new MyViewHolder(itemView);
    }

    @Override
    public void onBindViewHolder(MyViewHolder holder, int position) {
        holder.textViewName.setText(RecordList.get(position).getName());
        holder.textViewMove.setText(RecordList.get(position).getMove());
        holder.textViewRMove.setText(RecordList.get(position).getRmove());
        holder.textViewResult.setText(RecordList.get(position).getResult());

    }

    @Override
    public int getItemCount() {
        return RecordList.size();
    }


}


