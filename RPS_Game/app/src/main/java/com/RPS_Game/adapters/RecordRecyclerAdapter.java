package com.RPS_Game.adapters;


import android.support.v7.widget.AppCompatTextView;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import com.RPS_Game.model.Record;
import com.RPS_Game.tflite.R;

import java.util.List;



public class RecordRecyclerAdapter extends RecyclerView.Adapter<RecordRecyclerAdapter.UserViewHolder> {

    private List<Record> listRecords;

    public RecordRecyclerAdapter(List<Record> listUsers) {
        this.listRecords = listRecords;
    }

    @Override
    public UserViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        // inflating recycler item view
        View itemView = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_user_recycler, parent, false);

        return new UserViewHolder(itemView);
    }

    @Override
    public void onBindViewHolder(UserViewHolder holder, int position) {
        holder.textViewName.setText(listRecords.get(position).getName());
        holder.textViewMove.setText(listRecords.get(position).getMove());
        holder.textViewRMove.setText(listRecords.get(position).getRmove());
        holder.textViewResult.setText(listRecords.get(position).getRmove());
    }

    @Override
    public int getItemCount() {
//        Log.v(RecordRecyclerAdapter.class.getSimpleName(),""+listRecords.size());
//        return listRecords.size();
        return 10;
    }



    public class UserViewHolder extends RecyclerView.ViewHolder {

        public AppCompatTextView textViewName;
        public AppCompatTextView textViewMove;
        public AppCompatTextView textViewRMove;
        public AppCompatTextView textViewResult;

        public UserViewHolder(View view) {
            super(view);
            textViewName = (AppCompatTextView) view.findViewById(R.id.randomMove);
            textViewMove = (AppCompatTextView) view.findViewById(R.id.textViewMove);
            textViewRMove = (AppCompatTextView) view.findViewById(R.id.textViewRmove);
            textViewResult = (AppCompatTextView) view.findViewById(R.id.textResult);
        }
    }


}
