#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

struct Node {
    string name;
    string op;
    vector<int> inputs;
    vector<vector<float>> weights;
    vector<float> bias;
};

struct Graph {
    vector<Node> nodes;
};

vector<float> linear(const vector<float>& x,
                     const vector<vector<float>>& w,
                     const vector<float>& b) {
    vector<float> y(b.size());

    for (int i = 0; i < (int)b.size(); i++) {
        float sum = b[i];

        for (int j = 0; j < (int)x.size(); j++) {
            sum += w[i][j] * x[j];
        }

        y[i] = sum;
    }

    return y;
}

vector<float> relu(vector<float> x) {
    for (int i = 0; i < (int)x.size(); i++) {
        x[i] = max(0.0f, x[i]);
    }

    return x;
}

vector<float> execute(const Graph& graph, const vector<float>& input) {
    vector<vector<float>> values(graph.nodes.size());

    for (int i = 0; i < (int)graph.nodes.size(); i++) {
        const Node& node = graph.nodes[i];

        if (node.op == "Input") {
            values[i] = input;
        }
        else if (node.op == "Linear") {
            int src = node.inputs[0];
            values[i] = linear(values[src], node.weights, node.bias);
        }
        else if (node.op == "ReLU") {
            int src = node.inputs[0];
            values[i] = relu(values[src]);
        }
    }

    return values.back();
}

int main() {
    Graph graph;

    Node input;
    input.name = "input";
    input.op = "Input";
    graph.nodes.push_back(input);

    Node fc1;
    fc1.name = "fc1";
    fc1.op = "Linear";
    fc1.inputs = {0};

    fc1.weights = {
        {0.1, 0.2, 0.3, 0.4},
        {0.2, 0.1, 0.4, 0.3},
        {0.3, 0.4, 0.1, 0.2},
        {0.4, 0.3, 0.2, 0.1},
        {0.1, 0.1, 0.1, 0.1},
        {0.2, 0.2, 0.2, 0.2},
        {0.3, 0.3, 0.3, 0.3},
        {0.4, 0.4, 0.4, 0.4}
    };

    fc1.bias = {0, 0, 0, 0, 0, 0, 0, 0};
    graph.nodes.push_back(fc1);

    Node activation;
    activation.name = "relu1";
    activation.op = "ReLU";
    activation.inputs = {1};
    graph.nodes.push_back(activation);

    Node fc2;
    fc2.name = "fc2";
    fc2.op = "Linear";
    fc2.inputs = {2};

    fc2.weights = {
        {0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4},
        {0.4, 0.3, 0.2, 0.1, 0.4, 0.3, 0.2, 0.1}
    };

    fc2.bias = {0.1, 0.2};
    graph.nodes.push_back(fc2);

    cout << "Graph IR:" << endl;

    for (int i = 0; i < (int)graph.nodes.size(); i++) {
        cout << i << ": " << graph.nodes[i].name
             << " (" << graph.nodes[i].op << ")" << endl;
    }

    vector<float> x = {1, 2, 3, 4};
    vector<float> output = execute(graph, x);

    cout << "\nOutput:" << endl;

    for (float value : output) {
        cout << value << endl;
    }

    return 0;
}
