// --- IO ---
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.BufferedWriter;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileWriter;
import java.io.File;
import java.util.Comparator;

// --- Collections & utilities ---
import java.util.List;
import java.util.ArrayList;
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

// --- Folder management ---
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import java.awt.Image;
// --- Main ---
public class NotLearningDigitRecognizer {

    // ----------------------
    // Utilities / activations
    // ----------------------
    public static double relu(double x) { return Math.max(0, x); }
    public static double reluDeriv(double pre) { return pre > 0 ? 1.0 : 0.0; }

    public static double[] softmax(double[] x) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : x) if (v > max) max = v;
        double sum = 0.0;
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = Math.exp(x[i] - max);
            sum += out[i];
        }
        for (int i = 0; i < x.length; i++) out[i] /= sum;
        return out;
    }

    // ----------------------
    // Dataset helper
    // ----------------------
    public static class Dataset {
        public List<double[]> inputs = new ArrayList<>();
        public List<Integer> labels = new ArrayList<>();
    }

    /**
     * Load dataset from a path that may be a single CSV file or a directory containing CSV files.
     * Each row expected to have 785 columns: 0..783 = pixels, 784 = label (28*28 + 1).
     * Pixel values are normalized to [0,1] by dividing by 255.0.
     */
    public static Dataset loadDatasetFromPath(String path) throws IOException {
        final int EXPECTED_COLUMNS = 784 + 1; // pixels + label
        Dataset ds = new Dataset();
        File f = new File(path);
        if (!f.exists()) throw new FileNotFoundException("Path not found: " + path);

        List<File> csvFiles = new ArrayList<>();
        if (f.isFile()) {
            if (f.getName().toLowerCase().endsWith(".csv")) csvFiles.add(f);
            else throw new IOException("Provided file is not a .csv: " + path);
        } else {
            // Collect CSVs recursively from folder
            collectCSVFilesRecursive(f, csvFiles);
            if (csvFiles.isEmpty()) {
                System.out.println("No CSV files found in folder: " + path);
                return ds;
            }
            // sort for determinism
            Collections.sort(csvFiles, Comparator.comparing(File::getAbsolutePath));
        }

        for (File csv : csvFiles) {
            try (BufferedReader br = new BufferedReader(new FileReader(csv))) {
                String line = br.readLine();
                // Skip a header if it's obviously a header (contains 'pixel' or 'label')
                if (line != null && (line.toLowerCase().contains("pixel") || line.toLowerCase().contains("label"))) {
                    line = br.readLine();
                }
                int lineNo = 1;
                while (line != null) {
                    String[] parts = line.split(",");
                    if (parts.length != EXPECTED_COLUMNS) {
                        System.out.printf("Skipping invalid row in %s (line %d): length %d (expected %d)%n",
                                csv.getName(), lineNo, parts.length, EXPECTED_COLUMNS);
                        line = br.readLine(); lineNo++; continue;
                    }
                    double[] input = new double[784];
                    try {
                        for (int i = 0; i < 784; i++) input[i] = Double.parseDouble(parts[i]) / 255.0;
                        int label = Integer.parseInt(parts[784]);
                        ds.inputs.add(input);
                        ds.labels.add(label);
                    } catch (NumberFormatException nfe) {
                        System.out.printf("Skipping malformed numeric row in %s (line %d): %s%n", csv.getName(), lineNo, nfe.getMessage());
                    }
                    line = br.readLine(); lineNo++;
                }
            }
            System.out.println("Loaded from " + csv.getAbsolutePath());
        }

        System.out.println("Total samples loaded from path '" + path + "': " + ds.inputs.size());
        return ds;
    }

    private static void collectCSVFilesRecursive(File dir, List<File> out) {
        File[] files = dir.listFiles();
        if (files == null) return;
        for (File f : files) {
            if (f.isDirectory()) collectCSVFilesRecursive(f, out);
            else if (f.isFile() && f.getName().toLowerCase().endsWith(".csv")) out.add(f);
        }
    }

    // ----------------------
    // Conv block (2-layer advanced)
    // ----------------------
    static class ConvBlockAdvanced {
        int inH = 28, inW = 28;
        int kH = 3, kW = 3;
        int numFilters1, numFilters2;
        int conv1OutH, conv1OutW, pool1OutH, pool1OutW;
        int conv2OutH, conv2OutW, pool2OutH, pool2OutW;

        double[][][] K1, K2; // [F][kH][kW]
        double[] B1, B2;

        double[][] img2D;
        double[][][] conv1Pre, conv1Act, pooled1;
        int[][][] pool1MaxI, pool1MaxJ;
        double[][][] conv2Pre, conv2Act, pooled2;
        int[][][] pool2MaxI, pool2MaxJ;

        Random rnd;

        public ConvBlockAdvanced(int f1, int f2, Random rnd) {
            this.numFilters1 = f1; this.numFilters2 = f2; this.rnd = rnd;
            this.K1 = new double[f1][kH][kW]; this.B1 = new double[f1];
            this.K2 = new double[f2][kH][kW]; this.B2 = new double[f2];
            heInit(K1); heInit(K2);

            conv1OutH = inH - kH + 1; conv1OutW = inW - kW + 1;
            pool1OutH = conv1OutH / 2; pool1OutW = conv1OutW / 2;
            conv2OutH = pool1OutH - kH + 1; conv2OutW = pool1OutW - kW + 1;
            pool2OutH = conv2OutH / 2; pool2OutW = conv2OutW / 2;
        }

        private void heInit(double[][][] K) {
            double scale = Math.sqrt(2.0 / (kH * kW));
            for (int f = 0; f < K.length; f++)
                for (int i = 0; i < kH; i++)
                    for (int j = 0; j < kW; j++)
                        K[f][i][j] = rnd.nextGaussian() * scale;
        }

        public double[] forward(double[] input) {
            // reshape
            img2D = new double[inH][inW];
            for (int r = 0; r < inH; r++) for (int c = 0; c < inW; c++) img2D[r][c] = input[r * inW + c];
            // conv1
            conv1Pre = new double[numFilters1][conv1OutH][conv1OutW];
            conv1Act = new double[numFilters1][conv1OutH][conv1OutW];
            pooled1 = new double[numFilters1][pool1OutH][pool1OutW];
            pool1MaxI = new int[numFilters1][pool1OutH][pool1OutW];
            pool1MaxJ = new int[numFilters1][pool1OutH][pool1OutW];

            for (int f = 0; f < numFilters1; f++) {
                for (int i = 0; i < conv1OutH; i++) {
                    for (int j = 0; j < conv1OutW; j++) {
                        double s = B1[f];
                        for (int ki = 0; ki < kH; ki++)
                            for (int kj = 0; kj < kW; kj++)
                                s += img2D[i + ki][j + kj] * K1[f][ki][kj];
                        conv1Pre[f][i][j] = s;
                        conv1Act[f][i][j] = relu(s);
                    }
                }
            }
            // pool1 (2x2 max)
            for (int f = 0; f < numFilters1; f++) {
                for (int i = 0; i < pool1OutH; i++) {
                    for (int j = 0; j < pool1OutW; j++) {
                        int baseI = i * 2, baseJ = j * 2;
                        double mv = Double.NEGATIVE_INFINITY; int argI = baseI, argJ = baseJ;
                        for (int di = 0; di < 2; di++) for (int dj = 0; dj < 2; dj++) {
                            double v = conv1Act[f][baseI + di][baseJ + dj];
                            if (v > mv) { mv = v; argI = baseI + di; argJ = baseJ + dj; }
                        }
                        pooled1[f][i][j] = mv;
                        pool1MaxI[f][i][j] = argI;
                        pool1MaxJ[f][i][j] = argJ;
                    }
                }
            }

            // conv2 (treat pooled1 maps as "channels")
            conv2Pre = new double[numFilters2][conv2OutH][conv2OutW];
            conv2Act = new double[numFilters2][conv2OutH][conv2OutW];
            pooled2 = new double[numFilters2][pool2OutH][pool2OutW];
            pool2MaxI = new int[numFilters2][pool2OutH][pool2OutW];
            pool2MaxJ = new int[numFilters2][pool2OutH][pool2OutW];

            for (int f = 0; f < numFilters2; f++) {
                for (int i = 0; i < conv2OutH; i++) {
                    for (int j = 0; j < conv2OutW; j++) {
                        double s = B2[f];
                        // This simple mapping uses kernels across pooled1 feature maps
                        for (int ki = 0; ki < kH; ki++)
                            for (int kj = 0; kj < kW; kj++) {
                                int srcF = (ki * kW + kj) % Math.max(1, numFilters1);
                                int srcI = i + ki;
                                int srcJ = j + kj;
                                if (srcI >= 0 && srcI < pool1OutH && srcJ >= 0 && srcJ < pool1OutW)
                                    s += pooled1[srcF][srcI][srcJ] * K2[f][ki][kj];
                            }
                        conv2Pre[f][i][j] = s;
                        conv2Act[f][i][j] = relu(s);
                    }
                }
            }
            // pool2
            for (int f = 0; f < numFilters2; f++) {
                for (int i = 0; i < pool2OutH; i++) {
                    for (int j = 0; j < pool2OutW; j++) {
                        int baseI = i * 2, baseJ = j * 2;
                        double mv = Double.NEGATIVE_INFINITY; int argI = baseI, argJ = baseJ;
                        for (int di = 0; di < 2; di++) for (int dj = 0; dj < 2; dj++) {
                            double v = conv2Act[f][baseI + di][baseJ + dj];
                            if (v > mv) { mv = v; argI = baseI + di; argJ = baseJ + dj; }
                        }
                        pooled2[f][i][j] = mv;
                        pool2MaxI[f][i][j] = argI;
                        pool2MaxJ[f][i][j] = argJ;
                    }
                }
            }

            // flatten pooled2
            int outSize = numFilters2 * pool2OutH * pool2OutW;
            double[] feat = new double[outSize];
            int idx = 0;
            for (int f = 0; f < numFilters2; f++)
                for (int i = 0; i < pool2OutH; i++)
                    for (int j = 0; j < pool2OutW; j++)
                        feat[idx++] = pooled2[f][i][j];
            return feat;
        }

        /**
         * Backprop expects:
         * dFeat length = numFilters2 * pool2OutH * pool2OutW
         * accumulators dK1_acc, dB1_acc, dK2_acc, dB2_acc are provided by caller
         */
        public void backward(double[] dFeat, double[][][] dK1_acc, double[] dB1_acc,
                             double[][][] dK2_acc, double[] dB2_acc) {
            // Pool2 -> Conv2
            double[][][] dConv2Act = new double[numFilters2][conv2OutH][conv2OutW];
            int idx = 0;
            for (int f = 0; f < numFilters2; f++) {
                for (int i = 0; i < pool2OutH; i++) {
                    for (int j = 0; j < pool2OutW; j++) {
                        double grad = dFeat[idx++];
                        int mi = pool2MaxI[f][i][j], mj = pool2MaxJ[f][i][j];
                        dConv2Act[f][mi][mj] += grad;
                    }
                }
            }
            double[][][] dConv2Pre = new double[numFilters2][conv2OutH][conv2OutW];
            for (int f = 0; f < numFilters2; f++)
                for (int i = 0; i < conv2OutH; i++)
                    for (int j = 0; j < conv2OutW; j++)
                        dConv2Pre[f][i][j] = dConv2Act[f][i][j] * reluDeriv(conv2Pre[f][i][j]);

            // Gradients for K2,B2 and contribution to pooled1
            double[][][] dPooled1 = new double[numFilters1][pool1OutH][pool1OutW];
            for (int f = 0; f < numFilters2; f++) {
                for (int i = 0; i < conv2OutH; i++) {
                    for (int j = 0; j < conv2OutW; j++) {
                        double g = dConv2Pre[f][i][j];
                        for (int ki = 0; ki < kH; ki++) for (int kj = 0; kj < kW; kj++) {
                            int srcF = (ki * kW + kj) % Math.max(1, numFilters1);
                            int srcI = i + ki;
                            int srcJ = j + kj;
                            if (srcI >= 0 && srcI < pool1OutH && srcJ >= 0 && srcJ < pool1OutW) {
                                dK2_acc[f][ki][kj] += g * pooled1[srcF][srcI][srcJ];
                                dPooled1[srcF][srcI][srcJ] += g * K2[f][ki][kj];
                            }
                        }
                        dB2_acc[f] += g;
                    }
                }
            }

            // Pool1 -> Conv1
            double[][][] dConv1Act = new double[numFilters1][conv1OutH][conv1OutW];
            for (int f = 0; f < numFilters1; f++) {
                for (int i = 0; i < pool1OutH; i++) {
                    for (int j = 0; j < pool1OutW; j++) {
                        int mi = pool1MaxI[f][i][j], mj = pool1MaxJ[f][i][j];
                        dConv1Act[f][mi][mj] += dPooled1[f][i][j];
                    }
                }
            }
            double[][][] dConv1Pre = new double[numFilters1][conv1OutH][conv1OutW];
            for (int f = 0; f < numFilters1; f++)
                for (int i = 0; i < conv1OutH; i++)
                    for (int j = 0; j < conv1OutW; j++)
                        dConv1Pre[f][i][j] = dConv1Act[f][i][j] * reluDeriv(conv1Pre[f][i][j]);

            // grads for K1, B1
            for (int f = 0; f < numFilters1; f++) {
                for (int i = 0; i < conv1OutH; i++) {
                    for (int j = 0; j < conv1OutW; j++) {
                        double g = dConv1Pre[f][i][j];
                        dB1_acc[f] += g;
                        for (int ki = 0; ki < kH; ki++) for (int kj = 0; kj < kW; kj++) {
                            dK1_acc[f][ki][kj] += g * img2D[i + ki][j + kj];
                        }
                    }
                }
            }
        }
    } // end ConvBlockAdvanced

    // ----------------------
    // Dense forward/back cache
    // ----------------------
    static class DenseCache {
        double[] h1Raw, h1;
        double[] h2Raw, h2;
        double[] outRaw, out;
        double[] convFeat;
    }

    public static DenseCache denseForward(double[] input,
                                          double[][] W1, double[] b1,
                                          double[][] W2, double[] b2,
                                          double[][] W3, double[] b3,
                                          boolean training, double dropoutRate, Random rnd) {
        DenseCache c = new DenseCache();
        c.convFeat = input;

        int h1 = W1[0].length;
        int h2 = W2[0].length;
        int outSize = W3[0].length;

        c.h1Raw = new double[h1]; c.h1 = new double[h1];
        for (int j = 0; j < h1; j++) {
            double s = b1[j];
            for (int i = 0; i < input.length; i++) s += input[i] * W1[i][j];
            c.h1Raw[j] = s; c.h1[j] = relu(s);
        }
        if (training && dropoutRate > 0) {
            for (int j = 0; j < h1; j++) {
                if (rnd.nextDouble() < dropoutRate) c.h1[j] = 0.0;
                else c.h1[j] /= (1.0 - dropoutRate);
            }
        }

        c.h2Raw = new double[h2]; c.h2 = new double[h2];
        for (int j = 0; j < h2; j++) {
            double s = b2[j];
            for (int i = 0; i < h1; i++) s += c.h1[i] * W2[i][j];
            c.h2Raw[j] = s; c.h2[j] = relu(s);
        }
        if (training && dropoutRate > 0) {
            for (int j = 0; j < h2; j++) {
                if (rnd.nextDouble() < dropoutRate) c.h2[j] = 0.0;
                else c.h2[j] /= (1.0 - dropoutRate);
            }
        }

        c.outRaw = new double[outSize];
        for (int k = 0; k < outSize; k++) {
            double s = b3[k];
            for (int j = 0; j < h2; j++) s += c.h2[j] * W3[j][k];
            c.outRaw[k] = s;
        }
        c.out = softmax(c.outRaw);
        return c;
    }

    // ----------------------
    // Save / Load weights (objects)
    // ----------------------
    public static void saveWeightsObj(String filename,
                                      double[][][] K1, double[] B1,
                                      double[][][] K2, double[] B2,
                                      double[][] W1, double[] b1,
                                      double[][] W2, double[] b2,
                                      double[][] W3, double[] b3) throws IOException {
        try (ObjectOutputStream o = new ObjectOutputStream(new FileOutputStream(filename))) {
            o.writeObject(K1); o.writeObject(B1);
            o.writeObject(K2); o.writeObject(B2);
            o.writeObject(W1); o.writeObject(b1);
            o.writeObject(W2); o.writeObject(b2);
            o.writeObject(W3); o.writeObject(b3);
        }
    }

    @SuppressWarnings("unchecked")
    public static Object[] loadWeightsObj(String filename) throws IOException, ClassNotFoundException {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(filename))) {
            double[][][] K1 = (double[][][]) in.readObject();
            double[] B1 = (double[]) in.readObject();
            double[][][] K2 = (double[][][]) in.readObject();
            double[] B2 = (double[]) in.readObject();
            double[][] W1 = (double[][]) in.readObject(); double[] b1 = (double[]) in.readObject();
            double[][] W2 = (double[][]) in.readObject(); double[] b2 = (double[]) in.readObject();
            double[][] W3 = (double[][]) in.readObject(); double[] b3 = (double[]) in.readObject();
            return new Object[]{K1, B1, K2, B2, W1, b1, W2, b2, W3, b3};
        }
    }

    // ----------------------
    // Simple visualizer (existing)
    // ----------------------

    static class AccuracyVisualizer extends JFrame {
        AccuracyPanel totalAccPanel;
        AccuracyPanel[] digitPanels = new AccuracyPanel[10];

        public AccuracyVisualizer() {
            super("Live Accuracy Visualizer");
            setLayout(new GridLayout(2, 1));
            totalAccPanel = new AccuracyPanel("Total Accuracy");
            add(totalAccPanel);
            JPanel digitContainer = new JPanel(new GridLayout(2,5));
            for (int i = 0; i < 10; i++) {
                digitPanels[i] = new AccuracyPanel("Test for " + i);
                digitContainer.add(digitPanels[i]);
            }
            add(digitContainer);
            setSize(1200, 800);
            setLocationRelativeTo(null);
            setVisible(true);
        }
        public void update(int epoch, double totalAcc, double[] perDigitAcc){
            totalAccPanel.addPoint(epoch, totalAcc*100);
            for(int i=0;i<10;i++) digitPanels[i].addPoint(epoch, perDigitAcc[i]*100);
        }
        static class AccuracyPanel extends JPanel {
            java.util.List<java.awt.Point> pts = new ArrayList<>();
            java.util.List<Double> ys = new ArrayList<>();
            String title;
            public AccuracyPanel(String title){
                this.title = title;
                setBackground(Color.BLACK);
                setToolTipText("");
                addMouseMotionListener(new MouseMotionAdapter(){
                    public void mouseMoved(MouseEvent e){
                        int n = pts.size();
                        for(int i=0;i<n;i++){
                            java.awt.Point p = pts.get(i);
                            if (Math.abs(e.getX()-p.x) < 5 && Math.abs(e.getY()-p.y) < 5){
                                setToolTipText(String.format("%s: ep=%d, %.2f%%", title, i+1, ys.get(i)));
                                return;
                            }
                        }
                        setToolTipText(null);
                    }
                });
            }
            public void addPoint(double epoch, double val){ ys.add(val); repaint(); }
            @Override
            protected void paintComponent(Graphics g){
                super.paintComponent(g);
                Graphics2D g2=(Graphics2D)g;
                g2.setColor(Color.WHITE); g2.drawString(title, 10, 15);
                int w = getWidth(), h = getHeight();
                g2.setColor(Color.GRAY); g2.drawLine(40, h-30, w-10, h-30); g2.drawLine(40, h-30, 40, 10);
                int n = ys.size(); if (n==0) return;
                double maxY=100.0, minY=0.0; int plotW=w-60, plotH=h-50;
                pts.clear(); g2.setColor(Color.GREEN);
                for (int i=0;i<n;i++){
                    int x = 40 + (int)((i/(double)(Math.max(1, n-1)))*plotW);
                    int y = h-30 - (int)(((ys.get(i)-minY)/(maxY-minY))*plotH);
                    pts.add(new java.awt.Point(x,y));
                    g2.fillOval(x-2,y-2,4,4);
                    if (i>0) { java.awt.Point p = pts.get(i-1); g2.drawLine(p.x, p.y, x, y); }
                }
            }
        }
    }

    

    // ----------------------
    // MAIN training loop
    // ----------------------
    public static void main(String[] args) throws Exception {
        // Hyperparams
        int imgH = 28, imgW = 28;
        int inputSize = imgH * imgW; // 784
        int h1 = 256, h2 = 128, outputSize = 10;
        double lr = 0.01;
        int batchSize = 128;
        double lambda = 0.0001;
        int maxEpochs = 50; // reduce for quicker iteration; increase later
        double dropoutRate = 0.20;
        Random rnd = new Random(42);

        // Conv filters
        int numFilters1 = 8;
        int numFilters2 = 16;
        ConvBlockAdvanced conv = new ConvBlockAdvanced(numFilters1, numFilters2, rnd);

        // determine flattened feature size from conv object
        int flattened = conv.numFilters2 * conv.pool2OutH * conv.pool2OutW;

        // Dense params (He init)
        double[][] W1 = new double[flattened][h1]; double[] bb1 = new double[h1];
        double[][] W2 = new double[h1][h2]; double[] bb2 = new double[h2];
        double[][] W3 = new double[h2][outputSize]; double[] bb3 = new double[outputSize];

        for (int i=0;i<flattened;i++) for (int j=0;j<h1;j++) W1[i][j] = rnd.nextGaussian()*Math.sqrt(2.0/flattened);
        for (int j=0;j<h1;j++) bb1[j] = 0.0;
        for (int i=0;i<h1;i++) for (int j=0;j<h2;j++) W2[i][j] = rnd.nextGaussian()*Math.sqrt(2.0/h1);
        for (int j=0;j<h2;j++) bb2[j] = 0.0;
        for (int i=0;i<h2;i++) for (int j=0;j<outputSize;j++) W3[i][j] = rnd.nextGaussian()*Math.sqrt(2.0/h2);
        for (int j=0;j<outputSize;j++) bb3[j] = 0.0;

        // Adjust these to your actual folders or single csv files
        String trainPath = "C:\\Users\\danie\\Downloads\\Projects\\Data\\Neural Network\\CSV's\\Train";
        String testPath  = "C:\\Users\\danie\\Downloads\\Projects\\Data\\Neural Network\\CSV's\\Test";

        Dataset trainDs = loadDatasetFromPath(trainPath);
        Dataset testDs  = loadDatasetFromPath(testPath);

        List<double[]> trainInputs = trainDs.inputs;
        List<Integer> trainLabels  = trainDs.labels;
        List<double[]> testInputs  = testDs.inputs;
        List<Integer> testLabels   = testDs.labels;

        System.out.println("Train size: " + trainInputs.size() + "   Test size: " + testInputs.size());
        if (trainInputs.isEmpty()) { System.err.println("No training samples found - check paths and CSVs."); return; }

        AccuracyVisualizer viz = new AccuracyVisualizer();
        BufferedWriter logCsv = new BufferedWriter(new FileWriter("accuracy_logs.csv"));
        logCsv.write("epoch,totalAcc,d0,d1,d2,d3,d4,d5,d6,d7,d8,d9\n");

        // Training loop
        for (int epoch = 1; epoch <= maxEpochs; epoch++) {
            // shuffle
            List<Integer> idxs = new ArrayList<>();
            for (int i=0;i<trainInputs.size();i++) idxs.add(i);
            Collections.shuffle(idxs, rnd);

            for (int batchStart = 0; batchStart < trainInputs.size(); batchStart += batchSize) {
                int batchEnd = Math.min(batchStart + batchSize, trainInputs.size());
                int bs = batchEnd - batchStart;

                // accumulators
                double[][][] dK1 = new double[conv.numFilters1][conv.kH][conv.kW];
                double[] dB1_conv = new double[conv.numFilters1];
                double[][][] dK2 = new double[conv.numFilters2][conv.kH][conv.kW];
                double[] dB2_conv = new double[conv.numFilters2];

                double[][] dW1 = new double[flattened][h1]; double[] db1 = new double[h1];
                double[][] dW2 = new double[h1][h2]; double[] db2 = new double[h2];
                double[][] dW3 = new double[h2][outputSize]; double[] db3 = new double[outputSize];

                for (int p = batchStart; p < batchEnd; p++) {
                    int iSample = idxs.get(p);
                    double[] x = trainInputs.get(iSample);
                    int y = trainLabels.get(iSample);

                    double[] convFeat = conv.forward(x);
                    DenseCache cache = denseForward(convFeat, W1, bb1, W2, bb2, W3, bb3, true, dropoutRate, rnd);
                    double[] out = cache.out;

                    // softmax CE gradient
                    double[] dOut = new double[outputSize];
                    for (int k = 0; k < outputSize; k++) dOut[k] = out[k] - (k == y ? 1.0 : 0.0);

                    // layer3 grads
                    for (int k = 0; k < outputSize; k++) {
                        db3[k] += dOut[k];
                        for (int j = 0; j < h2; j++) dW3[j][k] += cache.h2[j] * dOut[k];
                    }
                    double[] dH2 = new double[h2];
                    for (int j = 0; j < h2; j++) {
                        double s = 0;
                        for (int k = 0; k < outputSize; k++) s += W3[j][k] * dOut[k];
                        dH2[j] = s * reluDeriv(cache.h2Raw[j]);
                    }

                    // layer2 grads
                    for (int j = 0; j < h2; j++) {
                        db2[j] += dH2[j];
                        for (int ii = 0; ii < h1; ii++) dW2[ii][j] += cache.h1[ii] * dH2[j];
                    }
                    double[] dH1 = new double[h1];
                    for (int ii = 0; ii < h1; ii++) {
                        double s = 0;
                        for (int j = 0; j < h2; j++) s += W2[ii][j] * dH2[j];
                        dH1[ii] = s * reluDeriv(cache.h1Raw[ii]);
                    }

                    // layer1 grads -> conv features
                    double[] dConvFeat = new double[flattened];
                    for (int j = 0; j < h1; j++) {
                        db1[j] += dH1[j];
                        for (int ii = 0; ii < flattened; ii++) {
                            dW1[ii][j] += convFeat[ii] * dH1[j];
                            dConvFeat[ii] += W1[ii][j] * dH1[j];
                        }
                    }

                    // conv backprop: update dK1/dB1 and dK2/dB2 accumulators
                    conv.backward(dConvFeat, dK1, dB1_conv, dK2, dB2_conv);
                } // end batch

                double invBs = 1.0 / Math.max(1, bs);

                // update conv layer1
                for (int f = 0; f < conv.numFilters1; f++) {
                    dB1_conv[f] *= invBs;
                    conv.B1[f] -= lr * dB1_conv[f];
                    for (int i = 0; i < conv.kH; i++) for (int j = 0; j < conv.kW; j++)
                        conv.K1[f][i][j] -= lr * (dK1[f][i][j] * invBs + lambda * conv.K1[f][i][j]);
                }
                // update conv layer2
                for (int f = 0; f < conv.numFilters2; f++) {
                    dB2_conv[f] *= invBs;
                    conv.B2[f] -= lr * dB2_conv[f];
                    for (int i = 0; i < conv.kH; i++) for (int j = 0; j < conv.kW; j++)
                        conv.K2[f][i][j] -= lr * (dK2[f][i][j] * invBs + lambda * conv.K2[f][i][j]);
                }

                // dense updates
                for (int i = 0; i < flattened; i++) for (int j = 0; j < h1; j++)
                    W1[i][j] -= lr * (dW1[i][j] * invBs + lambda * W1[i][j]);
                for (int j = 0; j < h1; j++) bb1[j] -= lr * (db1[j] * invBs);

                for (int i = 0; i < h1; i++) for (int j = 0; j < h2; j++)
                    W2[i][j] -= lr * (dW2[i][j] * invBs + lambda * W2[i][j]);
                for (int j = 0; j < h2; j++) bb2[j] -= lr * (db2[j] * invBs);

                for (int i = 0; i < h2; i++) for (int j = 0; j < outputSize; j++)
                    W3[i][j] -= lr * (dW3[i][j] * invBs + lambda * W3[i][j]);
                for (int j = 0; j < outputSize; j++) bb3[j] -= lr * (db3[j] * invBs);
            } // end iterate batches

            // evaluate on test set if available
            int correct = 0;
            double[] digitCorrect = new double[10], digitTotal = new double[10];
            if (!testInputs.isEmpty()) {
                for (int i = 0; i < testInputs.size(); i++) {
                    double[] feat = conv.forward(testInputs.get(i));
                    DenseCache c = denseForward(feat, W1, bb1, W2, bb2, W3, bb3, false, 0.0, rnd);
                    double[] out = c.out;
                    int pred = 0; double mx = out[0];
                    for (int k = 1; k < out.length; k++) if (out[k] > mx) { mx = out[k]; pred = k; }
                    int lab = testLabels.get(i);
                    digitTotal[lab]++; if (pred == lab) { correct++; digitCorrect[lab]++; }
                }
            }

            double acc = testInputs.isEmpty() ? Double.NaN : correct / (double) testInputs.size();
            double[] perDigitAcc = new double[10];
            for (int d = 0; d < 10; d++) perDigitAcc[d] = digitTotal[d] > 0 ? (digitCorrect[d] / digitTotal[d]) : 0.0;

            System.out.printf("Epoch %d: Total Accuracy %s%n", epoch, Double.isNaN(acc) ? "N/A" : String.format("%.2f%%", acc * 100.0));
            viz.update(epoch, Double.isNaN(acc) ? 0 : acc, perDigitAcc);
            logCsv.write(epoch + "," + (Double.isNaN(acc) ? "NaN" : Double.toString(acc)));
            for (int d = 0; d < 10; d++) logCsv.write("," + perDigitAcc[d]);
            logCsv.write("\n"); logCsv.flush();
        } // epochs

        logCsv.close();

        // save weights
        saveWeightsObj("weights.obj", conv.K1, conv.B1, conv.K2, conv.B2, W1, bb1, W2, bb2, W3, bb3);
        System.out.println("Training finished. Weights saved to weights.obj");
    }
}
