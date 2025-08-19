// --- IO ---
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.BufferedWriter;
import java.io.FileWriter;

// --- Collections & utilities ---
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.Arrays;
import java.util.Random;
import java.util.Collections;

// --- GUI / plotting ---
import javax.swing.JFrame;
import javax.swing.JPanel;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Color;
import java.awt.GridLayout;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;

// --- Main ---
public class DigitRecognizer {

    // =========================
    // Activation functions
    // =========================
    public static double relu(double x) { return Math.max(0, x); }
    public static double reluDerivativeFromPre(double pre) { return pre > 0 ? 1.0 : 0.0; }

    public static double[] softmax(double[] x) {
        double max = Arrays.stream(x).max().getAsDouble();
        double sum = 0.0;
        double[] exp = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            exp[i] = Math.exp(x[i] - max);
            sum += exp[i];
        }
        for (int i = 0; i < x.length; i++) exp[i] /= sum;
        return exp;
    }

    // =========================
    // Data loading (CSV)
    // =========================
    public static List<double[]> loadInputs(String filename, int inputSize) throws IOException {
        List<double[]> inputs = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line = br.readLine(); // skip header
            while ((line = br.readLine()) != null) {
                String[] tokens = line.split(",");
                double[] input = new double[inputSize];
                for (int i = 0; i < inputSize; i++) input[i] = Double.parseDouble(tokens[i]) / 255.0;
                inputs.add(input);
            }
        }
        return inputs;
    }

    public static List<Integer> loadLabels(String filename, int inputSize) throws IOException {
        List<Integer> labels = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line = br.readLine(); // skip header
            while ((line = br.readLine()) != null) {
                String[] tokens = line.split(",");
                labels.add(Integer.parseInt(tokens[inputSize]));
            }
        }
        return labels;
    }

    // =========================
    // Live Accuracy Visualizer
    // =========================
    static class AccuracyVisualizer extends JFrame {
        AccuracyPanel totalAccPanel;
        AccuracyPanel[] digitPanels = new AccuracyPanel[10];

        public AccuracyVisualizer() {
            super("Live Accuracy Visualizer");
            setLayout(new GridLayout(2, 1));

            totalAccPanel = new AccuracyPanel("Total Accuracy");
            add(totalAccPanel);

            JPanel digitContainer = new JPanel(new GridLayout(2,5));
            for(int i=0;i<10;i++){
                digitPanels[i] = new AccuracyPanel("Test for "+i);
                digitContainer.add(digitPanels[i]);
            }
            add(digitContainer);

            setSize(1200,800);
            setLocationRelativeTo(null);
            setVisible(true);
        }

        public void update(int epoch, double totalAcc, double[] perDigitAcc){
            totalAccPanel.addPoint(epoch, totalAcc*100);
            for(int i=0;i<10;i++){
                digitPanels[i].addPoint(epoch, perDigitAcc[i]*100);
            }
        }

        static class AccuracyPanel extends JPanel {
            java.util.List<java.awt.Point> pts = new ArrayList<>();
            java.util.List<Double> ys = new ArrayList<>();
            String title;
            public AccuracyPanel(String title){ 
                this.title=title;
                setBackground(Color.BLACK);
                setToolTipText("");
                addMouseMotionListener(new MouseMotionAdapter(){
                    public void mouseMoved(MouseEvent e){
                        // simple tooltip on nearest point
                        int n = pts.size();
                        for(int i=0;i<n;i++){
                            java.awt.Point p = pts.get(i);
                            if (Math.abs(e.getX()-p.x)<5 && Math.abs(e.getY()-p.y)<5){
                                setToolTipText(String.format("%s: ep=%d, %.2f%%", title, i+1, ys.get(i)));
                                return;
                            }
                        }
                        setToolTipText(null);
                    }
                });
            }
            public void addPoint(double epoch, double val){
                ys.add(val);
                repaint();
            }
            @Override
            protected void paintComponent(Graphics g){
                super.paintComponent(g);
                Graphics2D g2=(Graphics2D)g;
                g2.setColor(Color.WHITE);
                g2.drawString(title, 10, 15);
                int w = getWidth(), h = getHeight();
                g2.setColor(Color.GRAY);
                g2.drawLine(40, h-30, w-10, h-30);
                g2.drawLine(40, h-30, 40, 10);

                int n = ys.size();
                if (n==0) return;
                double maxY = 100.0, minY = 0.0;
                int plotW = w-60, plotH = h-50;
                pts.clear();
                g2.setColor(Color.GREEN);
                for (int i=0;i<n;i++){
                    int x = 40 + (int)((i/(double)(Math.max(1,n-1)))*plotW);
                    int y = h-30 - (int)(((ys.get(i)-minY)/(maxY-minY))*plotH);
                    pts.add(new java.awt.Point(x,y));
                    g2.fillOval(x-2,y-2,4,4);
                    if (i>0){
                        java.awt.Point p = pts.get(i-1);
                        g2.drawLine(p.x,p.y,x,y);
                    }
                }
            }
        }
    }

    // =========================
    // CNN layer (Conv + ReLU + MaxPool 2x2)
    // =========================
    static class ConvBlock {
        int inH=28, inW=28;
        int kH=3, kW=3;
        int numFilters;
        int convOutH = inH - 2; // 28-3+1=26
        int convOutW = inW - 2; // 26
        int poolKH=2, poolKW=2, poolStride=2;
        int poolOutH = convOutH/2; // 13
        int poolOutW = convOutW/2; // 13

        // Parameters
        double[][][] K;   // [F][kH][kW]
        double[] B;       // [F]

        // Caches for backprop
        double[][][] convPre;   // [F][26][26] pre-activation
        double[][][] convAct;   // [F][26][26] after ReLU
        double[][][] pooled;    // [F][13][13]
        int[][][] poolMaxI;     // [F][13][13] index of max (row in convAct)
        int[][][] poolMaxJ;     // [F][13][13] index of max (col in convAct)
        double[][] img2D;       // [28][28] input image cached

        Random rnd;

        public ConvBlock(int numFilters, Random rnd) {
            this.numFilters = numFilters;
            this.rnd = rnd;
            K = new double[numFilters][kH][kW];
            B = new double[numFilters];
            heInit();
        }

        private void heInit() {
            double scale = Math.sqrt(2.0/(kH*kW));
            for (int f=0; f<numFilters; f++) {
                for (int i=0;i<kH;i++){
                    for (int j=0;j<kW;j++){
                        K[f][i][j] = rnd.nextGaussian()*scale;
                    }
                }
                B[f] = 0.0;
            }
        }

        // forward pass: input 1D[784] -> caches + pooled
        public double[] forward(double[] input) {
            // reshape input to 28x28
            img2D = new double[inH][inW];
            for (int r=0;r<inH;r++)
                for (int c=0;c<inW;c++)
                    img2D[r][c] = input[r*inW + c];

            convPre = new double[numFilters][convOutH][convOutW];
            convAct = new double[numFilters][convOutH][convOutW];
            // conv + ReLU
            for (int f=0; f<numFilters; f++) {
                for (int i=0;i<convOutH;i++){
                    for (int j=0;j<convOutW;j++){
                        double s = B[f];
                        for (int ki=0; ki<kH; ki++) {
                            for (int kj=0; kj<kW; kj++) {
                                s += img2D[i+ki][j+kj]*K[f][ki][kj];
                            }
                        }
                        convPre[f][i][j] = s;
                        convAct[f][i][j] = relu(s);
                    }
                }
            }
            // maxpool 2x2
            pooled = new double[numFilters][poolOutH][poolOutW];
            poolMaxI = new int[numFilters][poolOutH][poolOutW];
            poolMaxJ = new int[numFilters][poolOutH][poolOutW];

            for (int f=0; f<numFilters; f++) {
                for (int i=0;i<poolOutH;i++){
                    for (int j=0;j<poolOutW;j++){
                        int baseI = i*poolStride;
                        int baseJ = j*poolStride;
                        double maxV = -1e18;
                        int argI = baseI, argJ = baseJ;
                        for (int di=0; di<poolKH; di++) {
                            for (int dj=0; dj<poolKW; dj++) {
                                double v = convAct[f][baseI+di][baseJ+dj];
                                if (v > maxV) { maxV = v; argI = baseI+di; argJ = baseJ+dj; }
                            }
                        }
                        pooled[f][i][j] = maxV;
                        poolMaxI[f][i][j] = argI;
                        poolMaxJ[f][i][j] = argJ;
                    }
                }
            }
            // flatten pooled -> vector
            double[] feat = new double[numFilters*poolOutH*poolOutW];
            int idx=0;
            for (int f=0; f<numFilters; f++)
                for (int i=0;i<poolOutH;i++)
                    for (int j=0;j<poolOutW;j++)
                        feat[idx++] = pooled[f][i][j];
            return feat;
        }

        // backward: take upstream gradient wrt flattened pooled output
        // returns nothing (we only need grads on K and B)
        public void backward(double[] dFeat,
                             double[][][] dK_acc, double[] dB_acc) {
            // unflatten dFeat -> dPooled
            double[][][] dPooled = new double[numFilters][poolOutH][poolOutW];
            int idx=0;
            for (int f=0; f<numFilters; f++)
                for (int i=0;i<poolOutH;i++)
                    for (int j=0;j<poolOutW;j++)
                        dPooled[f][i][j] = dFeat[idx++];

            // backprop through MaxPool: route to argmax
            double[][][] dConvAct = new double[numFilters][convOutH][convOutW];
            for (int f=0; f<numFilters; f++) {
                for (int i=0;i<poolOutH;i++){
                    for (int j=0;j<poolOutW;j++){
                        int mi = poolMaxI[f][i][j];
                        int mj = poolMaxJ[f][i][j];
                        dConvAct[f][mi][mj] += dPooled[f][i][j];
                    }
                }
            }

            // backprop ReLU: dConvPre = dConvAct * relu'(convPre)
            double[][][] dConvPre = new double[numFilters][convOutH][convOutW];
            for (int f=0; f<numFilters; f++) {
                for (int i=0;i<convOutH;i++){
                    for (int j=0;j<convOutW;j++){
                        dConvPre[f][i][j] = dConvAct[f][i][j] * reluDerivativeFromPre(convPre[f][i][j]);
                    }
                }
            }

            // grads for K and B
            for (int f=0; f<numFilters; f++) {
                double db = 0.0;
                for (int i=0;i<convOutH;i++){
                    for (int j=0;j<convOutW;j++){
                        double grad = dConvPre[f][i][j];
                        db += grad;
                        // kernel grads
                        for (int ki=0; ki<kH; ki++) {
                            for (int kj=0; kj<kW; kj++) {
                                dK_acc[f][ki][kj] += img2D[i+ki][j+kj] * grad;
                            }
                        }
                    }
                }
                dB_acc[f] += db;
            }
        }
    }

    // =========================
    // Dense forward helper
    // =========================
    static class DenseCache {
        double[] h1Raw, h1;
        double[] h2Raw, h2;
        double[] outRaw, out;
        double[] convFeat; // flattened features from conv block (for dropout scaling)
    }

    public static DenseCache denseForward(double[] input,
                                          double[][] w1, double[] b1,
                                          double[][] w2, double[] b2,
                                          double[][] w3, double[] b3,
                                          boolean training, double dropoutRate, Random rnd) {
        DenseCache cache = new DenseCache();
        cache.convFeat = input;

        int h1 = w1[0].length;
        int h2 = w2[0].length;
        int outputSize = w3[0].length;

        cache.h1Raw = new double[h1];
        cache.h1 = new double[h1];
        for (int j=0;j<h1;j++){
            double s = b1[j];
            for (int i=0;i<input.length;i++) s += input[i]*w1[i][j];
            cache.h1Raw[j]=s;
            cache.h1[j]=relu(s);
        }
        // dropout on h1
        if (training && dropoutRate>0) {
            for (int j=0;j<h1;j++){
                if (rnd.nextDouble() < dropoutRate) cache.h1[j] = 0.0;
                else cache.h1[j] /= (1.0 - dropoutRate); // inverted dropout
            }
        }

        cache.h2Raw = new double[h2];
        cache.h2 = new double[h2];
        for (int j=0;j<h2;j++){
            double s=b2[j];
            for (int i=0;i<h1;i++) s += cache.h1[i]*w2[i][j];
            cache.h2Raw[j]=s;
            cache.h2[j]=relu(s);
        }
        if (training && dropoutRate>0) {
            for (int j=0;j<h2;j++){
                if (rnd.nextDouble() < dropoutRate) cache.h2[j] = 0.0;
                else cache.h2[j] /= (1.0 - dropoutRate);
            }
        }

        cache.outRaw = new double[outputSize];
        for (int k=0;k<outputSize;k++){
            double s=b3[k];
            for (int j=0;j<h2;j++) s += cache.h2[j]*w3[j][k];
            cache.outRaw[k]=s;
        }
        cache.out = softmax(cache.outRaw);
        return cache;
    }

    // =========================
    // Weight save/load (includes conv)
    // =========================
    public static void saveWeights(String filename,
                                   double[][][] convK, double[] convB,
                                   double[][] w1, double[] b1,
                                   double[][] w2, double[] b2,
                                   double[][] w3, double[] b3) throws IOException {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename))) {
            out.writeObject(convK); out.writeObject(convB);
            out.writeObject(w1); out.writeObject(b1);
            out.writeObject(w2); out.writeObject(b2);
            out.writeObject(w3); out.writeObject(b3);
        }
    }

    @SuppressWarnings("unchecked")
    public static Object[] loadWeights(String filename) throws IOException, ClassNotFoundException {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(filename))) {
            double[][][] convK = (double[][][]) in.readObject();
            double[] convB = (double[]) in.readObject();
            double[][] w1 = (double[][]) in.readObject(); double[] b1 = (double[]) in.readObject();
            double[][] w2 = (double[][]) in.readObject(); double[] b2 = (double[]) in.readObject();
            double[][] w3 = (double[][]) in.readObject(); double[] b3 = (double[]) in.readObject();
            return new Object[]{convK, convB, w1, b1, w2, b2, w3, b3};
        }
    }

    // =========================
    // Training
    // =========================
    public static void main(String[] args) throws Exception {
        // ----- Hyperparams -----
        int imgH=28, imgW=28;
        int inputSize=imgH*imgW;
        int numFilters = 16;           // Conv filters
        int kH=3, kW=3;                // Conv kernel
        int convOutH = imgH - kH + 1;  // 26
        int convOutW = imgW - kW + 1;  // 26
        int poolOutH = convOutH/2;     // 13
        int poolOutW = convOutW/2;     // 13
        int flattened = numFilters * poolOutH * poolOutW; // 16*13*13 = 2704

        int h1=256, h2=128, outputSize=10;

        double lr=0.01;
        int batchSize=128;
        double lambda=0.0001;       // L2
        int maxEpochs=128;
        double dropoutRate=0.20;    // on dense layers

        Random rand=new Random(42);

        // ----- Parameters -----
        // Conv
        ConvBlock conv = new ConvBlock(numFilters, rand); // handles its own init

        // Dense
        double[][] w1=new double[flattened][h1]; double[] b1=new double[h1];
        double[][] w2=new double[h1][h2];         double[] b2=new double[h2];
        double[][] w3=new double[h2][outputSize]; double[] b3=new double[outputSize];

        // He init for dense
        for(int i=0;i<flattened;i++) for(int j=0;j<h1;j++) w1[i][j]=rand.nextGaussian()*Math.sqrt(2.0/flattened);
        for(int j=0;j<h1;j++) b1[j]=0;
        for(int i=0;i<h1;i++) for(int j=0;j<h2;j++) w2[i][j]=rand.nextGaussian()*Math.sqrt(2.0/h1);
        for(int j=0;j<h2;j++) b2[j]=0;
        for(int i=0;i<h2;i++) for(int j=0;j<outputSize;j++) w3[i][j]=rand.nextGaussian()*Math.sqrt(2.0/h2);
        for(int j=0;j<outputSize;j++) b3[j]=0;

        // ----- Data -----
        List<double[]> trainInputs=loadInputs("C:\\Users\\danie\\Downloads\\Projects\\Data\\Neural Network\\CSV's\\train.csv",inputSize);
        List<Integer> trainLabels=loadLabels("C:\\Users\\danie\\Downloads\\Projects\\Data\\Neural Network\\CSV's\\train.csv",inputSize);
        List<double[]> testInputs=loadInputs("C:\\Users\\danie\\Downloads\\Projects\\Data\\Neural Network\\CSV's\\test.csv",inputSize);
        List<Integer> testLabels=loadLabels("C:\\Users\\danie\\Downloads\\Projects\\Data\\Neural Network\\CSV's\\test.csv",inputSize);

        // ----- UI + CSV -----
        AccuracyVisualizer visualizer = new AccuracyVisualizer();
        BufferedWriter csv = new BufferedWriter(new FileWriter("accuracy_logs.csv"));
        csv.write("epoch,totalAcc,d0,d1,d2,d3,d4,d5,d6,d7,d8,d9\n");

        // ----- Training loop -----
        for(int epoch=1; epoch<=maxEpochs; epoch++){
            // Shuffle indices
            List<Integer> indices = new ArrayList<>();
            for(int i=0;i<trainInputs.size();i++) indices.add(i);
            Collections.shuffle(indices, rand);

            for (int batchStart=0; batchStart<trainInputs.size(); batchStart+=batchSize){
                int batchEnd=Math.min(batchStart+batchSize,trainInputs.size());
                int bs=batchEnd-batchStart;

                // Accumulators (Conv)
                double[][][] dK = new double[numFilters][kH][kW];
                double[] dB = new double[numFilters];

                // Accumulators (Dense)
                double[][] dW1=new double[flattened][h1]; double[] dB1=new double[h1];
                double[][] dW2=new double[h1][h2];       double[] dB2=new double[h2];
                double[][] dW3=new double[h2][outputSize]; double[] dB3=new double[outputSize];

                for (int p=batchStart; p<batchEnd; p++){
                    int idx = indices.get(p);
                    double[] x = trainInputs.get(idx);
                    int y = trainLabels.get(idx);

                    // ----- Forward: Conv -> Dense -----
                    double[] convFeat = conv.forward(x);
                    DenseCache cache = denseForward(convFeat, w1,b1, w2,b2, w3,b3, true, dropoutRate, rand);
                    double[] out = cache.out;

                    // ----- Loss grad (softmax CE): dOut = prob - onehot -----
                    double[] dOut = new double[outputSize];
                    for (int k=0;k<outputSize;k++) dOut[k] = out[k] - (k==y?1.0:0.0);

                    // ----- Dense backprop -----
                    // Layer 3 (h2 -> out)
                    for (int k=0;k<outputSize;k++){
                        dB3[k] += dOut[k];
                        for (int j=0;j<h2;j++) dW3[j][k] += cache.h2[j]*dOut[k];
                    }
                    // dH2
                    double[] dH2 = new double[h2];
                    for (int j=0;j<h2;j++){
                        double s=0;
                        for (int k=0;k<outputSize;k++) s += w3[j][k]*dOut[k];
                        dH2[j] = s * reluDerivativeFromPre(cache.h2Raw[j]);
                    }

                    // Layer 2 (h1 -> h2)
                    for (int j=0;j<h2;j++){
                        dB2[j] += dH2[j];
                        for (int i=0;i<h1;i++) dW2[i][j] += cache.h1[i]*dH2[j];
                    }
                    // dH1
                    double[] dH1 = new double[h1];
                    for (int i=0;i<h1;i++){
                        double s=0;
                        for (int j=0;j<h2;j++) s += w2[i][j]*dH2[j];
                        dH1[i] = s * reluDerivativeFromPre(cache.h1Raw[i]);
                    }

                    // Layer 1 (convFeat -> h1)
                    double[] dConvFeat = new double[flattened];
                    for (int j=0;j<h1;j++) {
                        dB1[j] += dH1[j];
                        for (int i=0;i<flattened;i++){
                            dW1[i][j] += convFeat[i]*dH1[j];
                            dConvFeat[i] += w1[i][j]*dH1[j];
                        }
                    }

                    // ----- Backprop into ConvBlock -----
                    conv.backward(dConvFeat, dK, dB);
                }

                // ----- Apply L2 and SGD updates -----
                double invBs = 1.0/bs;

                // Conv params
                for (int f=0; f<numFilters; f++){
                    dB[f] *= invBs;
                    conv.B[f] -= lr * dB[f];
                    for (int i=0;i<kH;i++){
                        for (int j=0;j<kW;j++){
                            double grad = dK[f][i][j]*invBs + lambda*conv.K[f][i][j];
                            conv.K[f][i][j] -= lr * grad;
                        }
                    }
                }

                // Dense params
                for(int i=0;i<flattened;i++) for(int j=0;j<h1;j++) w1[i][j]-=lr*(dW1[i][j]*invBs + lambda*w1[i][j]);
                for(int j=0;j<h1;j++) b1[j]-=lr*(dB1[j]*invBs);

                for(int i=0;i<h1;i++) for(int j=0;j<h2;j++) w2[i][j]-=lr*(dW2[i][j]*invBs + lambda*w2[i][j]);
                for(int j=0;j<h2;j++) b2[j]-=lr*(dB2[j]*invBs);

                for(int i=0;i<h2;i++) for(int j=0;j<outputSize;j++) w3[i][j]-=lr*(dW3[i][j]*invBs + lambda*w3[i][j]);
                for(int j=0;j<outputSize;j++) b3[j]-=lr*(dB3[j]*invBs);
            }

            // ----- Evaluate on test set -----
            int correct=0;
            double[] digitCorrect = new double[10];
            double[] digitTotal = new double[10];

            for (int i=0;i<testInputs.size();i++){
                // forward (no dropout)
                double[] convFeat = conv.forward(testInputs.get(i));
                DenseCache cache = denseForward(convFeat, w1,b1, w2,b2, w3,b3, false, 0.0, rand);
                double[] out = cache.out;

                int pred=0; double mx=out[0];
                for (int k=1;k<out.length;k++){ if (out[k]>mx){ mx=out[k]; pred=k; } }
                int label = testLabels.get(i);
                if (pred==label) correct++;
                digitTotal[label]++;
                if (pred==label) digitCorrect[label]++;
            }
            double acc = correct / (double) testInputs.size();
            double[] perDigitAcc = new double[10];
            for (int d=0; d<10; d++) perDigitAcc[d] = digitTotal[d] > 0 ? (digitCorrect[d]/digitTotal[d]) : 0.0;

            System.out.printf("Epoch %d: Total Accuracy %.2f%%%n", epoch, acc*100.0);
            // UI + CSV
            visualizer.update(epoch, acc, perDigitAcc);
            csv.write(epoch + "," + acc);
            for (int d=0; d<10; d++) csv.write("," + perDigitAcc[d]);
            csv.write("\n"); csv.flush();
        }

        csv.close();
        // Save weights
        saveWeights("weights.dat", 
            conv.K, conv.B,
            w1,b1,w2,b2,w3,b3);

        System.out.println("Training complete! Weights saved to weights.dat");
    }
}
