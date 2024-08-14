#include <bits/stdc++.h>
using namespace std;

#define debug(x) cerr << #x << " = " << x << "\n";
#define RAND(a, b) uniform_int_distribution<int>(a, b)(rng)
#define RANDR(a, b) uniform_real_distribution<double>(a, b)(rng)
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

//simulated fp16
class Float16{
    static const uint32_t mantissaShift = 42;
    static const uint32_t expShiftMid   = 56;
    static const uint32_t expShiftOut   = 52;
    double dValue_;

public:    
    Float16(double in) : dValue_(in) {
        uint64_t utmp;
        memcpy(&utmp, &dValue_, sizeof utmp);
        //zeroing mantissa bits starting from 11th (this is NOT rounding)
        utmp = utmp >> mantissaShift;
        utmp = utmp << mantissaShift;
        //setting masks for 5-bit exponent extraction out of 11-bit one
        const uint64_t maskExpMid = (63llu << expShiftMid);
        const uint64_t maskExpOut = (15llu << expShiftOut);
        const uint64_t maskExpLead = (1llu << 62);
        const uint64_t maskMantissaD = (1llu << 63) + maskExpLead + maskExpMid + maskExpOut;
        if (utmp & maskExpLead) {// checking leading bit, suspect overflow
            if (utmp & maskExpMid) { //Detected overflow if at least 1 bit is non-zero
                //Assign Inf with proper sign
                utmp = utmp | maskExpMid; //setting 1s in the middle 6 bits of of exponent
                utmp = utmp & maskMantissaD; //zeroing mantissa irrelative of original values to prevent NaN
                utmp = utmp | maskExpOut; //setting 1s in the last 4 bits of exponent
            }
        } else { //checking small numbers according to exponent range
            if ((utmp & maskExpMid) != maskExpMid) { //Detected underflow if at least 1 bit is 0
                utmp = 0;
            }
        }
        memcpy(&dValue_, &utmp, sizeof utmp);
    }

    Float16() : dValue_(0) {}

    Float16& operator=(const Float16& rhs) {
        this->dValue_ = rhs.dValue_;
        return *this;
    }

    Float16& operator=(const double& rhs) {
        this->dValue_ = rhs;
        uint64_t utmp;
        memcpy(&utmp, &dValue_, sizeof utmp);
        utmp = utmp >> mantissaShift;
        utmp = utmp << mantissaShift;
        memcpy(&dValue_, &utmp, sizeof utmp);
        return *this;
    }

    friend Float16 operator+(const Float16& lhs, const Float16& rhs) {
        double tmp = lhs.dValue_ + rhs.dValue_;
        return Float16(tmp);
    }

    float convert2Float() { return static_cast<float>(dValue_); }
    double convert2Double() { return dValue_; }
};

struct node {
    double realSum = 0.0;
    double proposedSum = 0.0;

    int type = 0;

    int cnt16 = 0;
    int cnt32 = 0;
    int cnt64 = 0;

    bool bad = true;
    int idx = -1;

    node () {}

    node (double v, int i) {
        realSum = proposedSum = v;
        bad = false;
        idx = i;
    }

    node (double rS, double pS, int t) {
        realSum = rS;
        pS = proposedSum;
        type = t;
    }
};

ostream& operator<<(ostream& os, const node &n) {
    os << "realSum: " << n.realSum << "\n";
    os << "proposedSum: " << n.proposedSum << "\n";
    os << "cnt16: " << n.cnt16 << "\n";
    os << "cnt32: " << n.cnt32 << "\n";
    os << "cnt64: " << n.cnt64 << "\n";
    os << "type: " << n.type << "\n";
    os << "bad: " << n.bad << "\n";
    os << "idx: " << n.idx << "\n";
    return os;
}

inline node combine (const node &a, const node &b, const int &type) {
    node result;

    if (a.bad && b.bad) {
        return result;
    }

    if (a.bad) {
        return b;
    }

    if (b.bad) {
        return a;
    }

    result.bad = false;
    result.idx = a.idx / 2;

    result.realSum = a.realSum + b.realSum;

    result.cnt16 = a.cnt16 + b.cnt16;
    result.cnt32 = a.cnt32 + b.cnt32;
    result.cnt64 = a.cnt64 + b.cnt64;

    result.type = type;

    double num1 = a.proposedSum;
    double num2 = b.proposedSum;
    Float16 f1(0), f2(0), res(0); 
    switch (type) {
        case 0:
            f1 = Float16(num1);
            f2 = Float16(num2);
            res = f1 + f2;
            result.proposedSum = res.convert2Double();
            result.cnt16++;
            break;
        case 1:
            result.proposedSum = static_cast<float>(num1) + static_cast<float>(num2);
            result.cnt32++;
            break;
        case 2:
            result.proposedSum = num1 + num2;
            result.cnt64++;
            break;
        default:
            cerr << "Unknown type\n";
    }

    return result;
}

inline void completeUpdate (vector<node> &solution) {
    int n = solution.size() >> 1;

    for (int i = n - 1; i > 0; i--) {
        solution[i] = combine(solution[i << 1], solution[i << 1 | 1], solution[i].type);
        assert((i << 1) != (i << 1 | 1));
    }
}

inline void singleUpdate (vector<node> &solution, int idx, const int &type) {
    int n = solution.size() >> 1;

    assert((idx << 1) != (idx << 1 | 1));
    solution[idx] = combine(solution[idx << 1], solution[idx << 1 | 1], type);
    for (; idx >>= 1;) {
        assert((idx << 1) != (idx << 1 | 1));
        solution[idx] = combine(solution[idx << 1], solution[idx << 1 | 1], solution[idx].type);
    }
}

inline string helperType (const int &type) {
    if (type == 0)
        return "h";
    if (type == 1)
        return "s";
    return "d";
}

string getAlgorithm (const vector<node> &solution, const int &n, int idx, const int &nn) {
    debug(idx);
    // debug(solution[idx].idx);
    // cerr << "===============\n";

    if (idx >= nn) {
        return to_string(idx - nn + 1);
    }
    // if (idx > n) {
    //     return to_string(idx - n + 1);
    // }
    
    // return "{" + helperType(solution[idx].type) + ":" + getAlgorithm(solution, n, idx << 1 | 1) + "," + getAlgorithm(solution, n, idx << 1) + "}";
    if (!solution[idx].bad)
        return "{" + helperType(solution[idx].type) + ":" + getAlgorithm(solution, n, idx << 1, nn) + "," + getAlgorithm(solution, n, idx << 1 | 1, nn) + "}";
    return "";
}

// string getAlgorithm(const vector<node> &solution, const int &n, int idx) {
//     struct Frame {
//         int idx;
//         string left;
//         string right;
//         bool visitedLeft;
//         bool visitedRight;
        
//         Frame(int i) : idx(i), visitedLeft(false), visitedRight(false) {}
//     };

//     stack<Frame> stack;
//     stack.push(Frame(1)); 
//     string result;

//     result.reserve(n * 10);  

//     while (!stack.empty()) {
//         Frame& current = stack.top();

//         if (current.idx >= n) {
//             result = to_string(current.idx - n + 1);
//             stack.pop();
//             if (!stack.empty()) {
//                 if (!stack.top().visitedLeft) {
//                     stack.top().left = result;
//                     stack.top().visitedLeft = true;
//                 } else {
//                     stack.top().right = result;
//                     stack.top().visitedRight = true;
//                 }
//             }
//         } else if (!current.visitedLeft) {
//             stack.push(Frame(current.idx << 1));
//         } else if (!current.visitedRight) {
//             stack.push(Frame(current.idx << 1 | 1));
//         } else {
//             result = "{";
//             result += helperType(solution[current.idx].type);
//             result += ":";
//             result += current.left;
//             result += ",";
//             result += current.right;
//             result += "}";
//             stack.pop();
//             if (!stack.empty()) {
//                 if (!stack.top().visitedLeft) {
//                     stack.top().left = result;
//                     stack.top().visitedLeft = true;
//                 } else {
//                     stack.top().right = result;
//                     stack.top().visitedRight = true;
//                 }
//             }
//         }
//     }

//     return result;
// }

// string imprimirEnSegmentTree(const vector<node>& solution, int n, int idx, int start, int end) {
//     if (end - start == 1) {
//         return to_string(start + 1);  // Imprimir la hoja correspondiente.
//     }

//     int mid = start + (end - start) / 2;
    
//     string izq = imprimirEnSegmentTree(solution, n, idx << 1, start, mid);
//     string der = imprimirEnSegmentTree(solution, n, idx << 1 | 1, mid, end);

//     return "{" + helperType(solution[idx].type) + ":" + izq + "," + der + "}";
// }

// string getAlgorithm(const vector<node>& solution, int n, int idx) {
//     return imprimirEnSegmentTree(solution, n, 1, 0, n);
// }

inline double getA (const vector<node> &solution) {
    double Se = solution[1].realSum;
    double Sc = solution[1].proposedSum;
    double eps1 = 1E-200;
    double eps2 = 1E-20;
    double expo = 0.05;

    double absDiff = abs(Sc - Se);
    double denom = max(abs(Se), eps1);
    double ratio = absDiff / denom;

    double maxVal = max(ratio, eps2);
    double A = pow(maxVal, expo);

    return A;
}

inline double getW (const vector<node> &solution) {
    node n = solution[1];

    int cnt16 = n.cnt16;
    int cnt32 = n.cnt32 * 2;
    int cnt64 = n.cnt64 * 4;

    double W = cnt16 + cnt32 + cnt64;

    return W;
}

// You must've updated the solution before
inline double fitness(const vector<node> &solution) {
    int n = solution.size() >> 1;

    double A = getA(solution);

    double W = getW(solution);
    double C = W / (n - 1);
    double D = 10.0 / sqrt(C + 0.5);

    double score = D / A;

    return score;
}

vector<int> randNumbers;
inline int roulette (int id) {
    double h = log2(id);
    
    double pH = 1 / (h + 1);
    double pD = h / (h + 1);
    // double pS = pH + pD;
    double pS = (pH + pD) / 2;

    double s = pH + pD + pS;

    vector<double> p = {pD, pS, pH};
    for (auto &x: p) {
        x /= s;
    }
    for (int i = 1; i < 3; i++) {
        p[i] += p[i - 1];
    }
    
    double r = RANDR(0.0, 1.0);
    for (int i = 0; i < 3; i++) {
        if (r <= p[i]) {
            return i;
        }
    }

    return 2;
}

/*
Update with a single point and random point
*/
// vector<node> genSol(const vector<node>& solution) {
//     int n = solution.size() >> 1;
//     vector<node> sol = solution;
 
//     int idx = RAND(1, n - 1);
//     singleUpdate(sol, idx, roulette(idx));
 
//     return sol;
// }

vector<pair<int, int>> genSol(vector<node>& solution) {
    int n = solution.size() >> 1;
    vector<pair<int, int>> changes;

    int idx = RAND(1, n - 1);
    while (solution[idx].bad && solution[idx].idx != -1) {
        idx = RAND(1, n - 1);
    }

    while (idx < n && solution[idx].idx != -1) {
        assert(!solution[idx].bad);
        double dontChange = 1;
        dontChange /= pow((double)log2(idx + 1), 0.5);
        
        if (RANDR(0.0, 1.0) <= dontChange) {
            changes.emplace_back(idx, solution[idx].type);
            singleUpdate(solution, idx, roulette(idx));
        }

        double rSide = RANDR(0.0, 1.0);
        int side = 0;

        if (rSide <= (double) 50.0) {
            side = 1;
        }
        
        idx = (idx << 1 | side);
    }
 
    return changes;
}

// vector<node> genSol(const vector<node>& solution) {
//     int n = solution.size() >> 1;
//     vector<node> sol = solution;

//     int lim = max(1, (int) log2(max(n - 1, 1)));
//     int cant = lim;
//     // int cant = RAND(1, n - 1);
//     unsigned seed = chrono::system_clock::now().time_since_epoch().count();
//     shuffle(randNumbers.begin(), randNumbers.end(), default_random_engine(seed));

//     for (int i = 0; i < cant; i++) {
//         int who = randNumbers[i];

//         double h = log2(who);
//         double p = 1 / pow(h + 1.0, 0.5);
//         double r = RANDR(0.0, 1.0);

//         if (r > p) {
//             sol[who].type = roulette(who);
//         }
//     }

//     completeUpdate(sol);

//     return sol;
// }   

vector<node> annealing (vector<node> &solution) {
    int n = solution.size() >> 1;
    double temperature = 10000.0;
    double alpha = 0.5;
    // int it = 10000;
    int it = 100;

    vector<node> bestSolution = solution;
    double bestFitness = fitness(solution);
    double currentFitness = bestFitness;

    // int accepted = 0;

    for (int i = 0; i < it; i++) {
        // auto start = std::chrono::high_resolution_clock::now();

        // auto newSol = genSol(solution);
        // double newFitness = fitness(newSol);

        auto changes = genSol(solution);
        reverse(changes.begin(), changes.end());
        double newFitness = fitness(solution);

        double delta = newFitness - currentFitness;

        if (newFitness > bestFitness) {
            bestFitness = newFitness;
            // bestSolution = newSol;
            bestSolution = solution;
        }

        if (delta > 0 || exp(delta / temperature) > RANDR(0.0, 1.0)) {
            currentFitness = newFitness;
            // accepted++;
            // solution = newSol;
            // debug(i);
            // debug(currentFitness);
            // debug(temperature);
        } else {
            for (const auto &[idx, type]: changes) {
                singleUpdate(solution, idx, type);
            }
        }

        temperature *= alpha;
    }

    debug(bestFitness);
    // debug(accepted);
    return bestSolution;
}

int powerTwo (int n) {
    int p = 1;
    while (p < n) {
        p <<= 1;
    }

    return p;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n;

    vector<double> nums(n);
    for (auto &x: nums) {
        cin >> x;
    }

    cout << fixed << setprecision(10);
    cerr << fixed << setprecision(10);

    // randNumbers.resize(n - 1);
    // iota(randNumbers.begin(), randNumbers.end(), 1);
    int nn = powerTwo(n);

    vector<node> st(nn << 1);
    for (int i = 0; i < n; i++) {
        st[i + nn] = node(nums[i], i + 1);
        st[i].type = 2;
    }

    completeUpdate(st);

    // for (auto x: st) {
    //     cerr << x << "\n";
    // }
    
    auto sol = annealing(st);

    for (auto x: sol) {
        cerr << x << "\n\n";
    }

    cout << getAlgorithm(sol, n, 1, nn) << "\n";

    return 0;
}

/*

{d:{h:{h:6,5},{h:4,3}},{d:{d:2,{h:6,5}},{d:{d:4,3},{d:2,1}}}}
{d:{d:{d:{d:1,2},{d:3,4}},{d:{h:5,6},2}},{h:{h:3,4},{h:5,6}}}

{d:{d:{d:{d:7,8},{d:9,10}},{d:1,2}},{h:{h:3,4},{h:5,6}}}


*/