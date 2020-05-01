"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (Object.hasOwnProperty.call(mod, k)) result[k] = mod[k];
    result["default"] = mod;
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
const mnist_1 = __importDefault(require("mnist"));
const math = __importStar(require("mathjs"));
const TRAINING_SET_SIZE = 1000;
const TEST_SET_SIZE = 1000;
const ALPHA = 0.03;
const ITERATIONS = 50;
const LAMBDA = 0.3;
main(TRAINING_SET_SIZE, TEST_SET_SIZE, ALPHA, ITERATIONS, LAMBDA);
function main(trainingSetSize, testSetSize, alpha, iterations, lambda) {
    // Part 0: Preparation
    console.log('Part 0: Preparation ...\n');
    const { training, test } = mnist_1.default.set(trainingSetSize, testSetSize);
    let X = training.map((v) => [...v.input]);
    let y = training.map((v) => [v.output.reverse().reduce(toDecimal, 0)]);
    let m = y.length;
    let n = X[0].length;
    console.log('Size of training set (m): ', m);
    console.log('Pixels in one picture (n): ', n);
    console.log('\n');
    // Part 1: Cost Function
    console.log('Part 1: Cost Function ...\n');
    // Add Intercept Term
    X = math.concat(Array(m)
        .fill(0)
        .map(() => [1]), X);
    let theta = Array(n + 1)
        .fill(0)
        .map(() => [0]);
    let { cost: untrainedCost } = costFunction(theta, X, y);
    console.log('Untrained Cost: ', untrainedCost);
    console.log('\n');
    // Part 2: Gradient Descent
    console.log('Part 2: Gradient Descent ...\n');
    const allTheta = oneVsAllGradientDescent(X, y, alpha, iterations, lambda);
    console.log('\n\n');
    // Part 3: Inference
    console.log('Part 3: Inference ...\n');
    let XTest = test.map((v) => [...v.input]);
    let yTest = test.map((v) => [v.output.reverse().reduce(toDecimal, 0)]);
    // let mTest = XTest.length;
    // Add Intercept Term
    XTest = math.concat(Array(m)
        .fill(0)
        .map(() => [1]), XTest);
    const predicted = predict(XTest, yTest, allTheta);
    const correctPredicted = predicted.filter((p) => p.predict === p.real).length;
    console.log(predicted);
    console.log(`${(correctPredicted / TEST_SET_SIZE) * 100} %`);
}
function sigmoid(z) {
    let g = math.evaluate(`1 ./ (1 + e.^-z)`, {
        z
    });
    return g;
}
function costFunction(theta, X, y) {
    const m = y.length;
    let h = sigmoid(math.evaluate(`X * theta`, {
        X,
        theta
    }));
    const cost = math.evaluate(`(1 / m) * (-y' * log(h) - (1 - y)' * log(1 - h))`, {
        h,
        y,
        m
    });
    const grad = math.evaluate(`(1 / m) * (h - y)' * X`, {
        h,
        y,
        m,
        X
    });
    return { cost: math.flatten(cost), grad };
}
function oneVsAllGradientDescent(X, y, ALPHA, ITERATIONS, LAMBDA) {
    const CLASSIFIERS = 10;
    const m = y.length;
    const n = X[0].length;
    const allTheta = [];
    for (let j = 0; j < CLASSIFIERS; j++) {
        // Note: Not n + 1, because X already has intercept term
        let theta = Array(n)
            .fill(0)
            .map(() => [0]);
        for (let i = 0; i < ITERATIONS; i++) {
            let h = sigmoid(math.evaluate(`X * theta`, {
                X,
                theta
            }));
            let regTheta = getRegularizedTheta(theta);
            let lambdaTerm = math.evaluate(`(lambda / m) * regTheta'`, {
                m,
                regTheta,
                lambda: LAMBDA
            });
            theta = math.evaluate(`theta - ALPHA / m * (((h - y)' * X) + lambdaTerm)'`, {
                theta,
                ALPHA,
                m,
                X,
                y: normalizeForOneVsAll(y, j),
                h,
                lambdaTerm
            });
            process.stdout.write(`Trained classifier ${j} of ${CLASSIFIERS} in iteration ${i} of ${ITERATIONS} \r`);
        }
        allTheta.push(math.flatten(theta));
    }
    return allTheta;
}
function predict(XTest, yTest, allTheta) {
    let raw = sigmoid(math.evaluate(`XTest * allTheta'`, {
        XTest,
        allTheta
    }));
    raw = raw.map((v) => v.reduce(maxProb, { p: 0, predict: -1 }));
    return raw.map((v, i) => (Object.assign(Object.assign({}, v), { real: yTest[i][0] })));
}
// Util
function maxProb(result, p, key) {
    if (result.p < p) {
        return { p, predict: key };
    }
    else {
        return result;
    }
}
function getRegularizedTheta(theta) {
    const [, ...shiftTheta] = theta;
    return [[0], ...shiftTheta];
}
function normalizeForOneVsAll(y, j) {
    // Note: y is an array of [[1], [9], [3], [5], ... ]
    // But we want [1] or [0] depending on classifier j
    return y.map((v) => {
        if (v[0] === j) {
            return [1];
        }
        else {
            return [0];
        }
    });
}
function toDecimal(result, value, key) {
    if (value) {
        return key;
    }
    else {
        return result;
    }
}
//# sourceMappingURL=index.js.map