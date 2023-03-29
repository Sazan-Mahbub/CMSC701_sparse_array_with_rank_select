#include <stdio.h>
#include <fstream>
#include <iostream>
#include <bitset>
#include <vector>
#include <cstdint>
#include <limits>
#include <sstream>
#include <iterator>
#include <cmath>
#include <algorithm>
#include<ctime>
#include <chrono>
#include <random>

/**
RESOURCES:
timestamp in C: https://stackoverflow.com/questions/59779870/timestamp-in-c
**/


using namespace std;
using namespace std::chrono;
using Clock = std::chrono::steady_clock;

class rank_support {

//private:
public:
    vector<bool> bitvector;
    vector<uint64_t> rank_map;
    uint64_t n;
    vector<uint64_t> jacobson_rank_bin;
    vector<vector<uint64_t>> jacobson_matrix;
    uint64_t chunk_size;
    uint64_t total_chunk_count;
    uint64_t sub_chunk_size;
    uint64_t sub_chunk_count;
    uint64_t n_elems;

//public:

    rank_support(){
    }

    rank_support(vector<bool> *bits_ref) {

        bitvector = *bits_ref;
        n = (*bits_ref).size();
        rank_map.resize(n + 1, 0);
        for (int i = 0; i < n; i++) {
            rank_map[i+1] = rank_map[i] + (*bits_ref)[i];
        }
        n_elems = rank_map[n];


        sub_chunk_size = floor(log2(n)/2);
        sub_chunk_count = ceil(2* log2(n));

        chunk_size = sub_chunk_count * sub_chunk_size;
        total_chunk_count = ceil((double)n/ (double)chunk_size);

        jacobson_rank_bin.resize(total_chunk_count, 0);

        jacobson_matrix.resize(total_chunk_count);
        for (int row = 0; row < total_chunk_count; row++) {
            jacobson_matrix[row].resize(sub_chunk_count);
        }

//        cout << (double)n/ (double)chunk_size<< "  "<<  n << " , bin size: " << chunk_size << ", bin count: " << total_chunk_count << endl;

        int chunk_counter = 0;
        int sub_chunk_counter = 0;
        uint64_t last_global_rank_track = 0;
        int position_of_chunk_start = 0;

        for (int i = 0; i < n; i++) {
            if ( (i % chunk_size) == 0 )
            {
                jacobson_rank_bin[chunk_counter] = rank_map[i];
                //last_global_rank_track = rank_map[i];
                chunk_counter++;
                sub_chunk_counter = 0;
                //position_of_chunk_start = i;
            }

            if ( (i% sub_chunk_size) == 0 )
            {
                jacobson_matrix[chunk_counter-1][sub_chunk_counter] = rank_map[i] - jacobson_rank_bin[chunk_counter-1];
                sub_chunk_counter++;
            }
        }
    }

    uint64_t rank1_jacobson(uint64_t i) { // jacobson's
        if (i >= n){
            return n_elems;
        }

        int chunk_counter = floor((double)i/ (double)chunk_size);
        int sub_chunk_counter = floor((double)(i - (chunk_counter*chunk_size)) / (double)sub_chunk_size);
        int start_pos = (chunk_counter * chunk_size) + (sub_chunk_counter * sub_chunk_size);
        return jacobson_rank_bin[chunk_counter] + jacobson_matrix[chunk_counter][sub_chunk_counter] + lookup (start_pos, i);
    }

    uint64_t lookup(uint64_t start_post, uint64_t i) {
        return count(bitvector.begin()+start_post, bitvector.begin()+i, true);
    }


    void print_chunk_ranks()
    {
        cout << "testing jacobson ";
        for (auto x: jacobson_rank_bin){
            cout << x << endl;
        }
    }

    uint64_t rank1(uint64_t i) { // precomputing and storing all ranks
        if (i >= n){
            return n_elems;
        }
        else{
            return rank_map[i]; // O(1) time, O(n) space.
        }
    }

    uint64_t overhead() {
        return (total_chunk_count + total_chunk_count*sub_chunk_count) * sizeof(uint64_t) * 8;
    }

    uint64_t overhead_more_space() {
        return (n+1) * sizeof(uint64_t) * 8;
    }

    void save(string& fname) {
        ofstream ofs(fname, ios::binary);
        ofs.write(reinterpret_cast<const char*>(&n), sizeof(n));
        ofs.write(reinterpret_cast<const char*>(&rank_map[0]), (n + 1) * sizeof(uint64_t));
        ofs.write(reinterpret_cast<const char*>(&n_elems), sizeof(n_elems));

        ofs.write(reinterpret_cast<const char*>(&sub_chunk_size), sizeof(sub_chunk_size));
        ofs.write(reinterpret_cast<const char*>(&sub_chunk_count), sizeof(sub_chunk_count));
        ofs.write(reinterpret_cast<const char*>(&chunk_size), sizeof(chunk_size));
        ofs.write(reinterpret_cast<const char*>(&total_chunk_count), sizeof(total_chunk_count));
        ofs.write(reinterpret_cast<const char*>(&jacobson_rank_bin[0]), (total_chunk_count) * sizeof(uint64_t));

        for (int i = 0; i < total_chunk_count; i++) {
            for (int j = 0; j < sub_chunk_count; j++) {
                ofs.write(reinterpret_cast<const char*>(&jacobson_matrix[i][j]), sizeof(jacobson_matrix[i][j]));
            }
        }
        ofs.close();
    }

    void load(string& fname) {
        ifstream ifs(fname, ios::binary);
        ifs.read(reinterpret_cast<char*>(&n), sizeof(n));
        rank_map.resize(n + 1);
        ifs.read(reinterpret_cast<char*>(&rank_map[0]), (n + 1) * sizeof(uint64_t));
        ifs.read(reinterpret_cast<char*>(&n_elems), sizeof(n_elems));

        ifs.read(reinterpret_cast<char*>(&sub_chunk_size), sizeof(sub_chunk_size));
        ifs.read(reinterpret_cast<char*>(&sub_chunk_count), sizeof(sub_chunk_count));
        ifs.read(reinterpret_cast<char*>(&chunk_size), sizeof(chunk_size));
        ifs.read(reinterpret_cast<char*>(&total_chunk_count), sizeof(total_chunk_count));

        jacobson_rank_bin.resize(total_chunk_count);
        jacobson_matrix.resize(total_chunk_count);
        for (int row = 0; row < total_chunk_count; row++) {
            jacobson_matrix[row].resize(sub_chunk_count);
        }
        ifs.read(reinterpret_cast<char*>(&jacobson_rank_bin[0]), (total_chunk_count) * sizeof(uint64_t));

        for (int i = 0; i < total_chunk_count; i++) {
            for (int j = 0; j < sub_chunk_count; j++) {
                ifs.read(reinterpret_cast<char*>(&jacobson_matrix[i][j]), sizeof(jacobson_matrix[i][j]));
            }
        }
        ifs.close();
    }
};

class select_support {
public:
    vector<uint64_t> select_map;
    uint64_t rank1_overhead;
    uint64_t ss_size;

    select_support() {
    }

    select_support(rank_support *rs_ref) {
        auto r = *rs_ref;
        ss_size = r.rank1(r.n);
        rank1_overhead = r.overhead();
        select_map.resize(ss_size, 0);

        int ss_index = 0;
        int max_rank = 0;
        for (int ii=0; ii<r.n+1; ii++){
            auto current_rank = r.rank1(ii);
            if (max_rank < current_rank){
//                cout << ii << "\t" << max_rank << " -- " << current_rank  << " --- " << select_map[ss_index]  << endl;
                select_map[ss_index] = ii - 1;
                max_rank = current_rank;
                ss_index += 1;
            }
        }
    }

    uint64_t select1(uint64_t i) {
        if (i >= ss_size){
            return select_map[ss_size-1];
        }
        else{
            return select_map[i]; // O(1) time, O(mlogn) space.
        }
    }

    uint64_t overhead_2() {
        return ss_size * sizeof(uint64_t) * 8;
    }

    uint64_t overhead() {
        return rank1_overhead + ss_size * sizeof(uint64_t) * 8;
    }

    void save(string& fname) {
        ofstream ofs(fname, ios::binary);
        ofs.write(reinterpret_cast<const char*>(&ss_size), sizeof(ss_size));
        ofs.write(reinterpret_cast<const char*>(&select_map[0]), ss_size * sizeof(uint64_t));
        ofs.close();
    }

    void load(string& fname) {
        ifstream ifs(fname, ios::binary);
        ifs.read(reinterpret_cast<char*>(&ss_size), sizeof(ss_size));
        select_map.resize(ss_size);
        ifs.read(reinterpret_cast<char*>(&select_map[0]), ss_size * sizeof(uint64_t));
        ifs.close();
    }
};


class sparse_array {

public:
    vector<bool> bitvector;
    vector<string> values;
    uint64_t n_bits = 0;
    uint64_t n_elems = 0;
    rank_support rs_obj;
    select_support ss_obj;
    bool is_finalized = false;
    uint64_t total_chars = 0;

    sparse_array() {
    }

    void create(uint64_t size) {
        if (!is_finalized) {
            bitvector.resize(size, 0);
            n_bits = size;
        }
        else{
            cout << "Sparse_array is already finalized. Cannot call create()." << endl;
        }
    }

    void append(string elem, uint64_t pos) {
        if (!is_finalized) {
            if (pos < n_bits) {
                bitvector[pos] = true;
                values.push_back(elem);
                total_chars += (elem.length() + 1);
                n_elems += 1;
            }
            else{
                cout << "Input position (" << pos << ") >= size_of_bitvector (" << n_bits << ")" << endl;
            }
        }
        else{
            cout << "Sparse_array is already finalized. Cannot call append()." << endl;
        }
    }

    void finalize() {
        rs_obj = rank_support(&bitvector);
        ss_obj = select_support(&rs_obj);
        is_finalized = true;
    }

    bool get_at_rank(uint64_t r, string& elem) {
        if (n_elems > r) { /// recheck again. r should not be equal to n_elems, but the hw2 description said "true" for "n_elems>=r".
            /*for r=n_elems, we need to take "values[n_elems-1]" instead*/
//            cout << r << "   " << n_elems << endl;
            elem = values[min(r, n_elems-1)]; /// recheck: is it getting the reference?
            return true;
        }
        return false;
    }

    bool get_at_index(uint64_t r, string& elem) {
        if ((!bitvector[r]) or (r >= n_bits)){
            return false;
        }
        elem = values[rs_obj.rank1(r)]; /// recheck: is it getting the reference?
        return true;
    }

    uint64_t get_index_of(uint64_t r) {
        if (r >= n_elems) {
            return numeric_limits<uint64_t>::max(); ///recheck: sentinel is not working?
        }
        return ss_obj.select1(r); /// recheck: index correct?
    }

    uint64_t num_elem_at(uint64_t r) {
        if (r >= n_bits) {
//            cout << r << " -- num_elem_at (r >= n_bits)  " << n_elems << endl;
            return n_elems;
        }
        return rs_obj.rank1(r+1);
    }

    uint64_t size() {
        return n_bits;
    }

    uint64_t num_elem() {
        return n_elems;
    }

    void save(string& fname) {
        ofstream ofs(string("sparse_array.").append(fname), ios::out | ios::binary);

        ofs.write(reinterpret_cast<const char*>(&n_bits), sizeof(n_bits));
        ofs.write(reinterpret_cast<const char*>(&n_elems), sizeof(n_elems));

        uint64_t str_len = 0;
        for (const auto& s: values){
            str_len = s.length();
            ofs.write(reinterpret_cast<const char*>(&str_len), sizeof(str_len));
            ofs.write(s.data(), str_len);
        }

        uint64_t bitvec_asint = 0;
        for (bool bit : bitvector) {
            bitvec_asint = (bitvec_asint << 1) | bit;
        }
//        cout << "\nsave --> bitvec_asint:" << bitvec_asint <<  endl;
        ofs.write(reinterpret_cast<const char*>(&bitvec_asint), sizeof(bitvec_asint));
        ofs.close();

        rs_obj.save(string("rank_support.").append(fname));
        ss_obj.save(string("select_support.").append(fname));
    }

    void load(string& fname) {
        ifstream ifs(string("sparse_array.").append(fname), ios::binary);

        ifs.read(reinterpret_cast<char*>(&n_bits), sizeof(n_bits));
        bitvector.resize(n_bits);
        ifs.read(reinterpret_cast<char*>(&n_elems), sizeof(n_elems));

        values = vector<string>();
        uint64_t str_len = 0;
        string s;
        for (int ii=0; ii<n_elems; ii++){
            ifs.read(reinterpret_cast<char*>(&str_len), sizeof(str_len));
            s = string(str_len, ' ');
            ifs.read(&s[0], str_len);
//            cout << s << endl;
            values.push_back(s);
        }

        uint64_t bitvec_asint;
        ifs.read(reinterpret_cast<char*>(&bitvec_asint), sizeof(bitvec_asint));
//        cout << "\nload --> bitvec_asint:" << bitvec_asint <<  endl;
        ifs.close();

        for (int ii=0; ii < n_bits; ii++){
            bitvector[n_bits - ii - 1] = (bitvec_asint & 1);
            bitvec_asint  = (bitvec_asint >> 1);
        }

        rs_obj.load(string("rank_support.").append(fname));
        ss_obj.load(string("select_support.").append(fname));

        is_finalized = true;
    }

    void print_data() {
        cout << "\nn_elems:" << n_elems;
        cout << "\nn_bits:" << n_bits;
        cout << "\nbitvector:\n\t";
        for (auto bit: bitvector){
            cout << bit;
        }
        cout << "\nelements:\n\t";
        for (auto elem: values){
            cout << elem << ", ";
        }
        cout << endl;

        cout << "\n\nrs_obj.n: " << rs_obj.n << endl;

        cout << "\nrank1():\n\t";
        for (int i=0; i<rs_obj.n+1; i++){
            cout << rs_obj.rank_map[i] << " ";
        }

        cout << "\njacobson_rank1():\n\t";
        for (int i=0; i<rs_obj.n+1; i++){
            cout << rs_obj.rank1(i) << " ";
        }

        cout << "\n\nss_obj.ss_size: " << ss_obj.ss_size << endl;

        cout << "\nselect1():\n\t";
        for (int i=0; i<ss_obj.ss_size; i++){
            cout << ss_obj.select1(i) << " ";
        }
        cout << endl;
    }
};

int main() {

    double zero_fraction = .5; /// fraction of the number of zeros in the bitvector
    int bitvec_size = 10000;

    vector<double> _zero_fraction_list_ = {0., .1, .3, .5, .7, .9, .95, .99, 1.};

    vector<int> _bitvect_sizes_ = {100, 500, 1000, 10000, 100000, 1000000};

    vector<double> create;
    vector<double> append;
    vector<double> finalize;
    vector<double> get_at_rank;
    vector<double> get_at_index;
    vector<double> get_index_of;
    vector<double> num_elem_at;
    vector<double> size;
    vector<double> num_elem;
    vector<double> save;
    vector<double> load;
    vector<double> total_time_list;

//    for (int bitvec_size: _bitvect_sizes_)
    for (double zero_fraction: _zero_fraction_list_)
    {
        double total_time_final = 0;

        cout << endl << endl;
        cout << "bitvec_size:\t" << bitvec_size << endl;
        cout << "zero_fraction\t" << zero_fraction << endl << endl;

        /// create sa
        auto t1 = Clock::now();
        double total_time = 0;
        for (int i=0; i<100; i++){
            sparse_array sa;
            t1 = Clock::now();
            sa.create(bitvec_size);
            total_time += (double) duration_cast<nanoseconds>(Clock::now() - t1).count()/1000;
        }
        cout << "sa.create():\t" << total_time/100 << " microseconds" <<  endl;
        total_time_final += total_time/100;
        create.push_back(total_time/100);

        sparse_array sa;
        sa.create(bitvec_size);

        /// adding elements
        vector<int> zerobits( (int) (bitvec_size * zero_fraction), 0);
        vector<int> onebits( bitvec_size - (int) (bitvec_size * zero_fraction), 1);

        vector<int> bits_flags(zerobits.begin(), zerobits.end());  /// note: this is not the bit vector used in the SA. this is just a vector of flags to make the insertion of "random elements" easy.
        bits_flags.insert(bits_flags.end(), onebits.begin(), onebits.end());

        random_shuffle(bits_flags.begin(), bits_flags.end());
        random_shuffle(bits_flags.begin(), bits_flags.end());

        vector<string> random_string = {"foo", "bar", "baz", "apple", "orange", "mango",
                                        "hello", "world", "sparse", "dense", "support", "test"};


        /// append()
        t1 = Clock::now();
        for (int i=0; i<bitvec_size; i++){
            if (bits_flags[i] == 1)
                sa.append(random_string[i % 12], i);
        }
        total_time = (double) duration_cast<nanoseconds>(Clock::now() - t1).count()/1000;
        cout << "sa.append():\t" << total_time/bitvec_size << " microseconds" <<  endl;
        total_time_final += total_time/bitvec_size;
        append.push_back(total_time/bitvec_size);


        /// finalize()
        t1 = Clock::now();
        for (int i=0; i<100; i++){
            sa.finalize();
        }
        total_time = (double) duration_cast<nanoseconds>(Clock::now() - t1).count()/1000;
        cout << "sa.finalize():\t" << total_time/100 << " microseconds" <<  endl;
        total_time_final += total_time/100;
        finalize.push_back(total_time/100);


        /// save()
        t1 = Clock::now();
        string fname = "sa.bin";
        for (int i=0; i<100; i++){
            sa.save(fname);
        }
        total_time = (double) duration_cast<nanoseconds>(Clock::now() - t1).count()/1000;
        cout << "sa.save():\t" << total_time/100 << " microseconds" <<  endl;
        total_time_final += total_time/100;
        save.push_back(total_time/100);



        /// load()
        sparse_array sa2;
        t1 = Clock::now();
        for (int i=0; i<100; i++){
            sa2.load(fname);
        }
        total_time = (double) duration_cast<nanoseconds>(Clock::now() - t1).count()/1000;
        cout << "sa.load():\t" << total_time/100 << " microseconds" <<  endl;
        total_time_final += total_time/100;
        load.push_back(total_time/100);


        /// get_at_rank()
        t1 = Clock::now();
        string elem;
        for (int i=0; i<100; i+=1){
            sa.get_at_rank(i%(sa.rs_obj.n+1), elem);
        }
        total_time = (double) duration_cast<nanoseconds>(Clock::now() - t1).count()/1000;
        cout << "sa.get_at_rank():\t" << total_time/100 << " microseconds" <<  endl;
        total_time_final += total_time/100;
        get_at_rank.push_back(total_time/100);



        /// get_at_index()
        t1 = Clock::now();
        for (int i=0; i<100; i+=1){
            sa.get_at_index(i%(sa.rs_obj.n), elem);
        }
        total_time = (double) duration_cast<nanoseconds>(Clock::now() - t1).count()/1000;
        cout << "sa.get_at_index():\t" << total_time/100 << " microseconds" <<  endl;
        total_time_final += total_time/100;
        get_at_index.push_back(total_time/100);



        /// get_index_of()
        t1 = Clock::now();
        for (int i=0; i<100; i+=1){
            sa.get_index_of(i%(sa.rs_obj.n+1));
        }
        total_time = (double) duration_cast<nanoseconds>(Clock::now() - t1).count()/1000;
        cout << "sa.get_index_of():\t" << total_time/100 << " microseconds" <<  endl;
        total_time_final += total_time/100;
        get_index_of.push_back(total_time/100);


        /// num_elem_at()
        t1 = Clock::now();
        for (int i=0; i<100; i+=1){
            sa.num_elem_at(i%(sa.rs_obj.n+1));
        }
        total_time = (double) duration_cast<nanoseconds>(Clock::now() - t1).count()/1000;
        cout << "sa.num_elem_at():\t" << total_time/100 << " microseconds" <<  endl;
        total_time_final += total_time/100;
        num_elem_at.push_back(total_time/100);


        /// size()
        t1 = Clock::now();
        for (int i=0; i<100; i+=1){
            sa.size();
        }
        total_time = (double) duration_cast<nanoseconds>(Clock::now() - t1).count()/1000;
        cout << "sa.size():\t" << total_time/100 << " microseconds" <<  endl;
        total_time_final += total_time/100;
        size.push_back(total_time/100);


        /// num_elem()
        t1 = Clock::now();
        for (int i=0; i<100; i+=1){
            sa.num_elem();
        }
        total_time = (double) duration_cast<nanoseconds>(Clock::now() - t1).count()/1000;
        cout << "sa.num_elem():\t" << total_time/100 << " microseconds" <<  endl;
        total_time_final += total_time/100;
        num_elem.push_back(total_time/100);

        cout << "total_time_final:\t" << total_time_final << " microseconds" <<  endl;
        total_time_list.push_back(total_time_final);


    //    sa.print_data();
    }

    cout << "create= [";
    for (auto x: create){
        cout << x <<  ", ";
    }
    cout << "]" << endl;

    cout << "append= [";
    for (auto x: append){
        cout << x <<  ", ";
    }
    cout << "]" << endl;

    cout << "finalize= [";
    for (auto x: finalize){
        cout << x <<  ", ";
    }
    cout << "]" << endl;

    cout << "get_at_rank= [";
    for (auto x: get_at_rank){
        cout << x <<  ", ";
    }
    cout << "]" << endl;

    cout << "get_at_index= [";
    for (auto x: get_at_index){
        cout << x <<  ", ";
    }
    cout << "]" << endl;

    cout << "get_index_of= [";
    for (auto x: get_index_of){
        cout << x <<  ", ";
    }
    cout << "]" << endl;

    cout << "num_elem_at= [";
    for (auto x: num_elem_at){
        cout << x <<  ", ";
    }
    cout << "]" << endl;

    cout << "size= [";
    for (auto x: size){
        cout << x <<  ", ";
    }
    cout << "]" << endl;

    cout << "num_elem= [";
    for (auto x: num_elem){
        cout << x <<  ", ";
    }
    cout << "]" << endl;

    cout << "save= [";
    for (auto x: save){
        cout << x <<  ", ";
    }
    cout << "]" << endl;

    cout << "load= [";
    for (auto x: load){
        cout << x <<  ", ";
    }
    cout << "]" << endl;

    cout << "total_time_list= [";
    for (auto x: total_time_list){
        cout << x <<  ", ";
    }
    cout << "]" << endl;

    return 0;
}
