# Flight-routing-system
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <queue>
#include <random>
#include <algorithm>
#include <iomanip>
#include <limits>

using namespace std;

constexpr double PI = 3.14159265358979323846;

double deg2rad(double deg){ return deg * PI / 180.0; }

double haversine_km(double lat1, double lon1, double lat2, double lon2){
    double R = 6371.0;
    double dlat = deg2rad(lat2 - lat1);
    double dlon = deg2rad(lon2 - lon1);
    double a = sin(dlat/2) * sin(dlat/2)
        + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dlon/2) * sin(dlon/2);
    double c = 2 * atan2(sqrt(a), sqrt(1 - a));
    return R * c;
}

// ----- Data structures -----
struct City {
    string name;
    double lat, lon;
    bool isHub;
    int assignedHubIdx;
    City(string n = "", double la = 0, double lo = 0, bool hub = false)
        : name(n), lat(la), lon(lo), isHub(hub), assignedHubIdx(-1) {}
};

struct Edge {
    int to;
    double baseKm;
    Edge(int t = 0, double d = 0) : to(t), baseKm(d) {}
};

struct DynamicFactors {
    double weather;   // 0..1
    double wind;      // 0..1
    double pressure;  // 0..1
    bool altitude;    // true/false
    DynamicFactors(double w = 0, double wi = 0, double p = 0, bool a = false)
        : weather(w), wind(wi), pressure(p), altitude(a) {}
};

// ----- Graph: hub-and-spoke -----
class HubGraph {
    vector<City> cities;
    vector<vector<Edge>> adj;
public:
    void addCity(const City &c){ cities.push_back(c); adj.emplace_back(); }
    int cityCount() const { return (int)cities.size(); }
    const vector<City>& getCities() const { return cities; }
    const vector<vector<Edge>>& getAdj() const { return adj; }

    int findCityIndex(const string &name) const {
        for (int i = 0; i < (int)cities.size(); ++i)
            if (cities[i].name == name) return i;
        return -1;
    }

    void connectBi(int a, int b){
        double d = haversine_km(cities[a].lat, cities[a].lon, cities[b].lat, cities[b].lon);
        adj[a].push_back(Edge(b, d));
        adj[b].push_back(Edge(a, d));
    }

    void assignCitiesToHubs(){
        vector<int> hubs;
        for (int i = 0; i < cityCount(); ++i)
            if (cities[i].isHub) hubs.push_back(i);

        for (int i = 0; i < cityCount(); ++i){
            if (cities[i].isHub) {
                cities[i].assignedHubIdx = i;
                continue;
            }
            double best = 1e18;
            int bh = -1;
            for (int h : hubs){
                double d = haversine_km(cities[i].lat, cities[i].lon, cities[h].lat, cities[h].lon);
                if (d < best){ best = d; bh = h; }
            }
            cities[i].assignedHubIdx = bh;
        }
    }

    void buildHubSpoke(){
        for (auto &v : adj) v.clear();
        assignCitiesToHubs();
        int n = cityCount();
        for (int i = 0; i < n; ++i){
            int h = cities[i].assignedHubIdx;
            if (h >= 0 && h != i) connectBi(i, h);
        }
        vector<int> hubs;
        for (int i = 0; i < n; ++i)
            if (cities[i].isHub) hubs.push_back(i);

        for (size_t i = 0; i < hubs.size(); ++i)
            for (size_t j = i + 1; j < hubs.size(); ++j)
                connectBi(hubs[i], hubs[j]);
    }

    void printSummary() const {
        cout << "\nNetwork summary:\n";
        for (int i = 0; i < cityCount(); ++i){
            cout << i << ": " << cities[i].name
                 << (cities[i].isHub ? " [HUB]" : "") << " -> ";
            for (auto &e : adj[i])
                cout << cities[e.to].name << "(" << (int)round(e.baseKm) << "km) ";
            cout << "\n";
        }
    }
};

// ----- Simple AI (weights applied to base distance) -----
struct Weights {
    double w_weather, w_wind, w_pressure, w_altitude;
    Weights(double a = 1.5, double b = 0.8, double c = 0.5, double d = 0.25)
        : w_weather(a), w_wind(b), w_pressure(c), w_altitude(d) {}
};

class SimpleAI {
    Weights w;
    double blockThreshold;
public:
    SimpleAI() : w(), blockThreshold(10000.0) {}
    void setWeights(const Weights &ww){ w = ww; }
    Weights getWeights() const { return w; }

    double adjustCost(double baseKm, const DynamicFactors &f) const {
        double factor = 1.0 + w.w_weather * f.weather + w.w_wind * f.wind + w.w_pressure * f.pressure;
        if (f.altitude) factor += w.w_altitude;
        return baseKm * factor;
    }

    bool blocked(double adjusted) const { return adjusted > blockThreshold; }
    void setBlockThreshold(double t){ blockThreshold = t; }
};

// ----- Dijkstra with adjusted weights -----
struct PathResult {
    vector<int> path;
    double totalCost;
    bool found;
};

PathResult dijkstraAdjusted(const HubGraph &g, int src, int dst,
                            const vector<vector<DynamicFactors>> &edgeFactors,
                            const SimpleAI &ai)
{
    int n = g.cityCount();
    const auto &adj = g.getAdj();
    vector<double> dist(n, numeric_limits<double>::infinity());
    vector<int> parent(n, -1);
    dist[src] = 0;

    using P = pair<double, int>;
    priority_queue<P, vector<P>, greater<P>> pq;
    pq.push({0, src});

    while (!pq.empty()){
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u]) continue;
        if (u == dst) break;

        for (int i = 0; i < (int)adj[u].size(); ++i){
            int v = adj[u][i].to;
            double base = adj[u][i].baseKm;
            DynamicFactors df;
            if (u < (int)edgeFactors.size() && i < (int)edgeFactors[u].size())
                df = edgeFactors[u][i];
            else
                df = DynamicFactors();

            double adjCost = ai.adjustCost(base, df);
            if (ai.blocked(adjCost)) continue;

            if (dist[v] > dist[u] + adjCost){
                dist[v] = dist[u] + adjCost;
                parent[v] = u;
                pq.push({dist[v], v});
            }
        }
    }

    PathResult res;
    if (dist[dst] >= numeric_limits<double>::infinity()) {
        res.found = false;
        res.totalCost = 0;
        return res;
    }

    res.found = true;
    res.totalCost = dist[dst];
    vector<int> rev;
    for (int cur = dst; cur != -1; cur = parent[cur])
        rev.push_back(cur);
    reverse(rev.begin(), rev.end());
    res.path = rev;
    return res;
}

// ----- Helpers to build default edgeFactors -----
vector<vector<DynamicFactors>> buildDefaultFactors(const HubGraph &g){
    int n = g.cityCount();
    vector<vector<DynamicFactors>> f(n);
    const auto &adj = g.getAdj();
    for (int i = 0; i < n; ++i)
        f[i].resize(adj[i].size(), DynamicFactors());
    return f;
}

// ----- Simulate realtime factors for all edges -----
void simulateRealtimeFactors(const HubGraph &g, vector<vector<DynamicFactors>> &edgeFactors,
                             double weather_strength = 0.3, double wind_strength = 0.3,
                             double pressure_strength = 0.2, double altitude_chance = 0.05)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> wdist(0.0, weather_strength);
    uniform_real_distribution<> windist(0.0, wind_strength);
    uniform_real_distribution<> pdist(0.0, pressure_strength);
    bernoulli_distribution alt(altitude_chance);

    int n = g.cityCount();
    const auto &adj = g.getAdj();
    for (int u = 0; u < n; ++u){
        for (int i = 0; i < (int)adj[u].size(); ++i){
            edgeFactors[u][i] = DynamicFactors(wdist(gen), windist(gen), pdist(gen), alt(gen));
        }
    }
}

// ----- Trainer: linear regression (least squares) -----
bool solveLinearSystem(vector<vector<double>> A, vector<double> b, vector<double> &x){
    int n = (int)b.size();
    const double EPS = 1e-12;

    for (int i = 0; i < n; ++i){
        int piv = i;
        for (int r = i + 1; r < n; ++r)
            if (fabs(A[r][i]) > fabs(A[piv][i])) piv = r;

        if (fabs(A[piv][i]) < EPS) return false;
        if (piv != i) {
            swap(A[piv], A[i]);
            swap(b[piv], b[i]);
        }

        double diag = A[i][i];
        for (int c = i; c < n; ++c) A[i][c] /= diag;
        b[i] /= diag;

        for (int r = 0; r < n; ++r){
            if (r == i) continue;
            double factor = A[r][i];
            for (int c = i; c < n; ++c) A[r][c] -= factor * A[i][c];
            b[r] -= factor * b[i];
        }
    }
    x = b;
    return true;
}

Weights trainWeightsSynthetic(int samples = 500, double noise_std = 0.02){
    double true_w1 = 1.4;
    double true_w2 = 0.9;
    double true_w3 = 0.45;
    double true_w4 = 0.2;

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> uf(0.0, 1.0);
    normal_distribution<> nnoise(0.0, noise_std);

    int m = samples, p = 4;
    vector<vector<double>> XT_X(p, vector<double>(p, 0.0));
    vector<double> XT_y(p, 0.0);

    for (int i = 0; i < m; ++i){
        double weather = uf(gen);
        double wind = uf(gen);
        double pressure = uf(gen);
        double altitude = (uf(gen) < 0.1) ? 1.0 : 0.0;

        double y = true_w1 * weather + true_w2 * wind + true_w3 * pressure + true_w4 * altitude + nnoise(gen);

        vector<double> row = {weather, wind, pressure, altitude};
        for (int a = 0; a < p; ++a)
            for (int b = 0; b < p; ++b)
                XT_X[a][b] += row[a] * row[b];
        for (int a = 0; a < p; ++a)
            XT_y[a] += row[a] * y;
    }

    vector<double> sol;
    bool ok = solveLinearSystem(XT_X, XT_y, sol);
    if (!ok){
        cout << "Trainer: failed to solve normal equations; returning default weights.\n";
        return Weights();
    }
    Weights learned(sol[0], sol[1], sol[2], sol[3]);
    return learned;
}

// ----- Pretty printing for a path -----
void printPath(const HubGraph &g, const vector<int> &path, double cost){
    if (path.empty()){
        cout << "No path\n";
        return;
    }
    const auto &cities = g.getCities();
    cout << "Path: ";
    for (size_t i = 0; i < path.size(); ++i){
        cout << cities[path[i]].name;
        if (i + 1 < path.size()) cout << " -> ";
    }
    cout << "\nTotal adjusted distance: " << fixed << setprecision(2) << cost << " km\n";
}

// ----- Build a sample network -----
HubGraph buildSampleNetwork(){
    HubGraph g;
    g.addCity(City("Delhi", 28.7041, 77.1025, true));
    g.addCity(City("Mumbai", 19.0760, 72.8777, true));
    g.addCity(City("Bengaluru", 12.9716, 77.5946, true));
    g.addCity(City("Chennai", 13.0827, 80.2707, true));

    g.addCity(City("Agra", 27.1767, 78.0081, false));
    g.addCity(City("Jaipur", 26.9124, 75.7873, false));
    g.addCity(City("Pune", 18.5204, 73.8567, false));
    g.addCity(City("Mysore", 12.2958, 76.6394, false));
    g.addCity(City("Coimbatore", 11.0168, 76.9558, false));

    g.buildHubSpoke();
    return g;
}

// ----- Main interactive program -----
int main(){
    cout << "=== Hub-based Flight Routing: simulation + trainer + AI ===\n\n";

    HubGraph g = buildSampleNetwork();
    g.printSummary();

    auto edgeFactors = buildDefaultFactors(g);

    cout << "\nWould you like to run the trainer to learn AI weights from synthetic data? (y/n): ";
    char doTrain;
    cin >> doTrain;

    SimpleAI ai;
    if (doTrain == 'y' || doTrain == 'Y'){
        cout << "Training (this uses synthetic examples and solves linear least squares)...\n";
        Weights learned = trainWeightsSynthetic(800, 0.02);
        ai.setWeights(learned);
        auto w = ai.getWeights();
        cout << fixed << setprecision(4)
             << "Learned weights: weather=" << w.w_weather << ", wind=" << w.w_wind
             << ", pressure=" << w.w_pressure << ", altitude=" << w.w_altitude << "\n";
    } else {
        cout << "Using default AI weights.\n";
    }

    cout << "\nSimulate random real-time factors across all edges? (y/n): ";
    char sim;
    cin >> sim;
    if (sim == 'y' || sim == 'Y'){
        simulateRealtimeFactors(g, edgeFactors, 0.7, 0.6, 0.4, 0.08);
        cout << "Simulated dynamic factors applied.\n";
    } else {
        cout << "Keeping factors at default (all-clear).\n";
    }

    cout << "\n>>> Running a sample batch of routes (base vs AI-adjusted):\n";
    vector<pair<string, string>> samplePairs = {
        {"Agra", "Pune"},
        {"Jaipur", "Coimbatore"},
        {"Pune", "Mysore"},
        {"Delhi", "Chennai"}
    };

    for (auto &pr : samplePairs){
        int s = g.findCityIndex(pr.first);
        int d = g.findCityIndex(pr.second);
        if (s < 0 || d < 0) continue;

        auto baseFactors = buildDefaultFactors(g);
        PathResult baseR = dijkstraAdjusted(g, s, d, baseFactors, ai);
        cout << "\nRoute: " << pr.first << " -> " << pr.second << "\n";
        if (baseR.found) printPath(g, baseR.path, baseR.totalCost);
        else cout << "  (no base route)\n";

        PathResult aiR = dijkstraAdjusted(g, s, d, edgeFactors, ai);
        if (aiR.found) {
            cout << "AI-adjusted route:\n";
            printPath(g, aiR.path, aiR.totalCost);
        } else {
            cout << "AI-adjusted: route blocked / no available path\n";
        }
    }

    while (true){
        cout << "\nEnter origin (or 'list' to show cities, 'exit' to quit): ";
        string origin;
        cin >> ws;
        getline(cin, origin);

        if (origin == "exit") break;

        if (origin == "list"){
            const auto &cities = g.getCities();
            cout << "\nCities:\n";
            for (int i = 0; i < (int)cities.size(); ++i)
                cout << " - " << cities[i].name << (cities[i].isHub ? " [HUB]" : "") << "\n";
            continue;
        }

        int s = g.findCityIndex(origin);
        if (s < 0){
            cout << "Unknown city\n";
            continue;
        }

        cout << "Enter destination: ";
        string dest;
        cin >> ws;
        getline(cin, dest);
        int d = g.findCityIndex(dest);
        if (d < 0){
            cout << "Unknown city\n";
            continue;
        }

        cout << "Re-simulate realtime factors for this run? (y/n): ";
        char rs;
        cin >> rs;
        if (rs == 'y' || rs == 'Y')
            simulateRealtimeFactors(g, edgeFactors, 0.6, 0.5, 0.3, 0.06);

        auto baseFactors = buildDefaultFactors(g);
        PathResult baseR = dijkstraAdjusted(g, s, d, baseFactors, ai);
        if (baseR.found) {
            cout << "\nBase route:\n";
            printPath(g, baseR.path, baseR.totalCost);
        } else {
            cout << "Base route: not found\n";
        }

        PathResult aiR = dijkstraAdjusted(g, s, d, edgeFactors, ai);
        if (aiR.found) {
            cout << "\nAI-adjusted route:\n";
            printPath(g, aiR.path, aiR.totalCost);
        } else {
            cout << "AI-adjusted route: blocked / not found\n";
        }

        cout << "\nAnother query? (y/n): ";
        char again;
        cin >> again;
        if (!(again == 'y' || again == 'Y')) break;
    }

    cout << "\nProgram finished.\n";
    return 0;
}
