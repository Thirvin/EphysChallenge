#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <random>
#include <tuple>
#include <vector>
using namespace std;

struct GivenInfo {
    int a, b;
    double W;
} w101[5050];

struct Collection {
    short distribution[101];  // 1 = A, -1 = B
    double E = 0;
} best;

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());  // Random Number Generator

inline void Input_w101() {
    ifstream fin("w101.txt");
    for (int i = 0; i < 5050; ++i) {
        fin >> w101[i].a >> w101[i].b >> w101[i].W;
        // change into 0-based
        --w101[i].a;
        --w101[i].b;
    }
    fin.close();
    return;
}

inline void OutputBest() {
    ofstream fout;
    fout.open("E.txt");
    fout << fixed << setprecision(4) << best.E;
    fout.close();

    fout.open("distribution.txt");
    for (short i = 0; i < 101; ++i) {
        if (best.distribution[i] == 1)
            fout << 'A';
        else
            fout << 'B';

        if (i != 100)
            fout << ' ';
    }
    fout.close();
    return;
}

inline void RandomAssign(Collection &cur) {
    // fixed the first student to class A
    cur.distribution[0] = 1;
    // randomly assign
    for (short i = 1; i < 101; ++i) {
        if (rng() % 2)
            cur.distribution[i] = 1;
        else
            cur.distribution[i] = -1;
    }
    return;
}

inline void RecordBest(Collection &cur) {
    if (best.E < cur.E) {
        best.E = cur.E;
        for (int i = 0; i < 101; ++i)
            best.distribution[i] = cur.distribution[i];
    }
    return;
}

class SimulatedAnnealing {
   public:
    double Initial_Temperature;
    double End_Temperature;
    double Alpha;
    int Round_Times;

    void run() {
        int cur_round = 0;
        while (cur_round++ < Round_Times) {
            Collection cur;
            Initialize(cur);
            double cur_temperature = Initial_Temperature;
            while (cur_temperature > End_Temperature) {
                // find a neighboring solution:
                // randomly select a student and switch to another class
                short position_of_switch = unif(rng);
                cur.distribution[position_of_switch] = -cur.distribution[position_of_switch];

                // calculate E after switch
                double new_E = cur.E;
                CalculateNewE(position_of_switch, new_E);

                double delta_E = new_E - cur.E;
                if (delta_E > 0)
                    cur.E = new_E;
                else if (exp(delta_E / cur_temperature) > (double)rng() / rng.max())
                    cur.E = new_E;
                else {  // not accept, switch back
                    cur.distribution[position_of_switch] = -cur.distribution[position_of_switch];
                    for (short i = 0; i < 101; ++i) {
                        edge[position_of_switch][i] = -edge[position_of_switch][i];
                        edge[i][position_of_switch] = -edge[i][position_of_switch];
                    }
                }
                // reduce temperature
                cur_temperature *= Alpha;
            }
            // record the best E and its distribution
            RecordBest(cur);
        }
        return;
    }

   private:
    uniform_int_distribution<int> unif = uniform_int_distribution<int>(1, 100);
    double edge[101][101];
    inline void Initialize(Collection &cur) {
        RandomAssign(cur);
        // calculate the first E and initialize edge
        for (short i = 0; i < 5050; ++i) {
            double tmp = w101[i].W * cur.distribution[w101[i].a] * cur.distribution[w101[i].b];
            cur.E += tmp;
            edge[w101[i].a][w101[i].b] = tmp;
            edge[w101[i].b][w101[i].a] = tmp;
        }
        return;
    }

    inline void CalculateNewE(short loc, double &new_E) {
        for (short i = 0; i < 101; ++i) {
            new_E -= edge[loc][i] * 2;
            edge[loc][i] = -edge[loc][i];
            edge[i][loc] = -edge[i][loc];
        }
        return;
    }
};

class GeneticAlgorithm {
   public:
    int Max_Generation;
    int Population_Size;
    double Mutation_Rate;
    int Offspring_Quantity;
    int Round_Times;

    void run() {
        population.resize(Population_Size);
        int cur_round = 0;
        while (cur_round++ < Round_Times) {
            Initialize();
            int generation = 0;
            while (generation++ < Max_Generation) {
                for (int i = 0; i < Offspring_Quantity / 2; ++i) {
                    int father, mather;
                    tie(father, mather) = ParentSelection();
                    Crossover(father, mather);
                }
                sort(population.begin(), population.end(), [&](const Collection &a, const Collection &b) { return a.E > b.E; });

                // survivor selection
                for (int i = 0; i < Offspring_Quantity; ++i)
                    population.pop_back();
            }
            RecordBest(population[0]);
        }
        return;
    }

   private:
    vector<Collection> population;

    void CountFitness(Collection &cur) {
        cur.E = 0;
        for (int i = 0; i < 5050; ++i)
            cur.E += w101[i].W * cur.distribution[w101[i].a] * cur.distribution[w101[i].b];
        return;
    }

    void Initialize() {
        for (int i = 0; i < Population_Size; ++i) {
            RandomAssign(population[i]);
            CountFitness(population[i]);
        }
        return;
    }

    inline pair<int, int> ParentSelection() {
        uniform_int_distribution<int> unif(0, Population_Size - 1);
        const int n = Population_Size / 10;
        int tmp[n];
        for (int i = 0; i < n; ++i)
            tmp[i] = unif(rng);

        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                if (population[tmp[i]].E < population[tmp[j]].E)
                    swap(tmp[i], tmp[j]);

        return {tmp[0], tmp[1]};
    }

    inline void Mutation(Collection &cur) {
        for (int i = 1; i < 101; ++i)
            if ((double)rng() / rng.max() < Mutation_Rate)
                cur.distribution[i] = -cur.distribution[i];
        return;
    }

    inline void Crossover(int father, int mather) {
        Collection son1, son2;
        for (int i = 0; i < 101; ++i) {
            if (rng() % 2) {
                son1.distribution[i] = population[father].distribution[i];
                son2.distribution[i] = population[mather].distribution[i];
            }
            else {
                son1.distribution[i] = population[mather].distribution[i];
                son2.distribution[i] = population[father].distribution[i];
            }
        }
        Mutation(son1);
        Mutation(son2);
        CountFitness(son1);
        CountFitness(son2);
        population.emplace_back(son1);
        population.emplace_back(son2);
        return;
    }
};

int main() {
    Input_w101();

    SimulatedAnnealing SA;
    SA.Initial_Temperature = 1e2;
    SA.End_Temperature = 1e-2;
    SA.Alpha = 0.9999;  // the rate of reducing temperature
    SA.Round_Times = 100;
    SA.run();

    GeneticAlgorithm GA;
    GA.Max_Generation = 1e4;
    GA.Population_Size = 100;
    GA.Mutation_Rate = 8.0 / 100;
    GA.Offspring_Quantity = 28;
    GA.Round_Times = 20;
    GA.run();

    // output the final ans
    OutputBest();

    return 0;
}