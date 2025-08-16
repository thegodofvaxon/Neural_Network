// --- IO ---
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.awt.event.MouseWheelListener;
import java.awt.event.MouseWheelEvent;

// --- Collections & utilities ---
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.Arrays;
import java.util.Random;
import java.util.Collections;

// --- GUI / plotting (if you want visualization later) ---
import javax.swing.JFrame;
import javax.swing.JPanel;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.RenderingHints;

// --- Optional GUI components ---
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.SwingUtilities;
import javax.swing.BorderFactory;
// --- GUI/visualizeer ---
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;
import java.awt.event.*;

    public class DigitRecognizer {

    // --- Activation functions ---
    public static double relu(double x) { return Math.max(0, x); }
    public static double reluDerivative(double x) { return x > 0 ? 1 : 0; }

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

    // --- Forward pass ---
    public static Map<String, Object> forward(double[] input, double[][] w1, double[] b1,
                                              double[][] w2, double[] b2,
                                              double[][] w3, double[] b3, boolean training) {
        int h1 = w1[0].length;
        int h2 = w2[0].length;
        int outputSize = w3[0].length;

        double[] hidden1Raw = new double[h1];
        double[] hidden1 = new double[h1];
        for (int j = 0; j < h1; j++) {
            double sum = b1[j];
            for (int i = 0; i < input.length; i++) sum += input[i] * w1[i][j];
            hidden1Raw[j] = sum;
            hidden1[j] = relu(sum);
        }

        double dropoutRate = 0.2;
        if (training) {
            Random rand = new Random();
            for (int j = 0; j < h1; j++) if (rand.nextDouble() < dropoutRate) hidden1[j] = 0;
        }

        double[] hidden2Raw = new double[h2];
        double[] hidden2 = new double[h2];
        for (int j = 0; j < h2; j++) {
            double sum = b2[j];
            for (int i = 0; i < h1; i++) sum += hidden1[i] * w2[i][j];
            hidden2Raw[j] = sum;
            hidden2[j] = relu(sum);
        }

        if (training) {
            Random rand = new Random();
            for (int j = 0; j < h2; j++) if (rand.nextDouble() < dropoutRate) hidden2[j] = 0;
        }

        double[] outputRaw = new double[outputSize];
        for (int k = 0; k < outputSize; k++) {
            double sum = b3[k];
            for (int j = 0; j < h2; j++) sum += hidden2[j] * w3[j][k];
            outputRaw[k] = sum;
        }

        double[] output = softmax(outputRaw);

        Map<String, Object> res = new HashMap<>();
        res.put("hidden1", hidden1); res.put("hidden1Raw", hidden1Raw);
        res.put("hidden2", hidden2); res.put("hidden2Raw", hidden2Raw);
        res.put("output", output); res.put("outputRaw", outputRaw);
        return res;
    }

    // --- CSV loading ---
    public static List<double[]> loadInputs(String filename, int inputSize) throws IOException {
        List<double[]> inputs = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(filename));
        String line = br.readLine(); // skip header
        while ((line = br.readLine()) != null) {
            String[] tokens = line.split(",");
            double[] input = new double[inputSize];
            for (int i = 0; i < inputSize; i++) input[i] = Double.parseDouble(tokens[i]) / 255.0;
            inputs.add(input);
        }
        br.close();
        return inputs;
    }

    public static List<Integer> loadLabels(String filename, int inputSize) throws IOException {
        List<Integer> labels = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(filename));
        String line = br.readLine(); // skip header
        while ((line = br.readLine()) != null) {
            String[] tokens = line.split(",");
            labels.add(Integer.parseInt(tokens[inputSize]));
        }
        br.close();
        return labels;
    }

    // --- Save/Load weights ---
    public static void saveWeights(String filename, double[][] w1, double[] b1,
                                   double[][] w2, double[] b2,
                                   double[][] w3, double[] b3) throws IOException {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename))) {
            out.writeObject(w1); out.writeObject(b1);
            out.writeObject(w2); out.writeObject(b2);
            out.writeObject(w3); out.writeObject(b3);
        }
    }

    public static Object[] loadWeights(String filename) throws IOException, ClassNotFoundException {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(filename))) {
            double[][] w1 = (double[][]) in.readObject(); double[] b1 = (double[]) in.readObject();
            double[][] w2 = (double[][]) in.readObject(); double[] b2 = (double[]) in.readObject();
            double[][] w3 = (double[][]) in.readObject(); double[] b3 = (double[]) in.readObject();
            return new Object[]{w1, b1, w2, b2, w3, b3};
        }
    }

    // --- Live Accuracy Visualizer ---
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
            java.util.List<Point.Double> points = new ArrayList<>();
            java.util.List<String> tooltips = new ArrayList<>();
            String title;
            double scaleX=1, scaleY=1;
            int offsetX=50, offsetY=30;

            public AccuracyPanel(String title){ 
                this.title=title;
                setBackground(Color.BLACK);
                setToolTipText(""); // enable tooltip
                addMouseMotionListener(new MouseMotionAdapter(){
                    public void mouseMoved(MouseEvent e){
                        for(int i=0;i<points.size();i++){
                            int x=(int)(points.get(i).x*scaleX)+offsetX;
                            int y=(int)(points.get(i).y*scaleY)+offsetY;
                            if(Math.abs(e.getX()-x)<5 && Math.abs(e.getY()-y)<5){
                                setToolTipText(tooltips.get(i));
                                return;
                            }
                        }
                        setToolTipText(null);
                    }
                });
                addMouseWheelListener(e->{
                    int rot=e.getWheelRotation();
                    if(rot>0){ scaleX*=1.1; scaleY*=1.1; }
                    else{ scaleX/=1.1; scaleY/=1.1; }
                    repaint();
                });
            }

            public void addPoint(double x, double y){
                points.add(new Point.Double(x,y));
                tooltips.add(String.format("Epoch %.0f, %.2f%%",x,y));
                repaint();
            }

            @Override
            protected void paintComponent(Graphics g){
                super.paintComponent(g);
                Graphics2D g2=(Graphics2D)g;
                g2.setColor(Color.WHITE);
                g2.drawString(title, getWidth()/2-50, 20);
                // axes
                g2.setColor(Color.GRAY);
                g2.drawLine(offsetX,getHeight()-offsetY,getWidth()-10,getHeight()-offsetY);
                g2.drawLine(offsetX,getHeight()-offsetY,offsetX,10);
                // draw points
                g2.setColor(Color.GREEN);
                for(int i=0;i<points.size();i++){
                    int x=(int)(points.get(i).x*scaleX)+offsetX;
                    int y=getHeight()-(int)(points.get(i).y*scaleY)-offsetY;
                    g2.fillOval(x-3,y-3,6,6);
                    if(i>0){
                        int prevX=(int)(points.get(i-1).x*scaleX)+offsetX;
                        int prevY=getHeight()-(int)(points.get(i-1).y*scaleY)-offsetY;
                        g2.drawLine(prevX,prevY,x,y);
                    }
                }
            }
        }
    }

    // --- Training ---
    public static void main(String[] args) throws Exception{
        int inputSize=28*28;
        int h1=256,h2=128;
        int outputSize=10;
        double lr=0.01;
        int batchSize=128;
        double lambda=0.0001;
        int maxEpochs=128;

        Random rand=new Random();
        double[][] w1=new double[inputSize][h1]; double[] b1=new double[h1];
        double[][] w2=new double[h1][h2]; double[] b2=new double[h2];
        double[][] w3=new double[h2][outputSize]; double[] b3=new double[outputSize];

        for(int i=0;i<inputSize;i++) for(int j=0;j<h1;j++) w1[i][j]=rand.nextGaussian()*Math.sqrt(2.0/inputSize);
        for(int j=0;j<h1;j++) b1[j]=0;
        for(int i=0;i<h1;i++) for(int j=0;j<h2;j++) w2[i][j]=rand.nextGaussian()*Math.sqrt(2.0/h1);
        for(int j=0;j<h2;j++) b2[j]=0;
        for(int i=0;i<h2;i++) for(int j=0;j<outputSize;j++) w3[i][j]=rand.nextGaussian()*Math.sqrt(2.0/h2);
        for(int j=0;j<outputSize;j++) b3[j]=0;

        List<double[]> trainInputs=loadInputs("Projects\\Data\\Neural Network\\CSV's\\train.csv",inputSize);
        List<Integer> trainLabels=loadLabels("Projects\\Data\\Neural Network\\CSV's\\train.csv",inputSize);
        List<double[]> testInputs=loadInputs("Projects\\Data\\Neural Network\\CSV's\\test.csv",inputSize);
        List<Integer> testLabels=loadLabels("Projects\\Data\\Neural Network\\CSV's\\test.csv",inputSize);

        AccuracyVisualizer visualizer = new AccuracyVisualizer();

        // Open CSV for logging
        BufferedWriter csv = new BufferedWriter(new FileWriter("accuracy_logs.csv"));
        csv.write("epoch,totalAcc,d0,d1,d2,d3,d4,d5,d6,d7,d8,d9\n");

        for(int epoch=1;epoch<=maxEpochs;epoch++){
            // Shuffle
            List<Integer> indices = new ArrayList<>();
            for(int i=0;i<trainInputs.size();i++) indices.add(i);
            Collections.shuffle(indices, rand);

            for(int batchStart=0; batchStart<trainInputs.size(); batchStart+=batchSize){
                int batchEnd=Math.min(batchStart+batchSize,trainInputs.size());
                int bs=batchEnd-batchStart;

                double[][] dW1=new double[inputSize][h1]; double[] dB1=new double[h1];
                double[][] dW2=new double[h1][h2]; double[] dB2=new double[h2];
                double[][] dW3=new double[h2][outputSize]; double[] dB3=new double[outputSize];

                for(int idx=batchStart;idx<batchEnd;idx++){
                    int i = indices.get(idx);
                    double[] x=trainInputs.get(i);
                    int y=trainLabels.get(i);

                    Map<String,Object> fwd=forward(x,w1,b1,w2,b2,w3,b3,true);
                    double[] h1v=(double[])fwd.get("hidden1");
                    double[] h1Raw=(double[])fwd.get("hidden1Raw");
                    double[] h2v=(double[])fwd.get("hidden2");
                    double[] h2Raw=(double[])fwd.get("hidden2Raw");
                    double[] out=(double[])fwd.get("output");

                    double[] dOut=new double[outputSize];
                    for(int k=0;k<outputSize;k++) dOut[k]=out[k]-(k==y?1:0);

                    for(int k=0;k<outputSize;k++){
                        dB3[k]+=dOut[k];
                        for(int j=0;j<h2;j++) dW3[j][k]+=h2v[j]*dOut[k];
                    }

                    double[] dH2=new double[h2];
                    for(int j=0;j<h2;j++){
                        double sum=0;
                        for(int k=0;k<outputSize;k++) sum+=w3[j][k]*dOut[k];
                        dH2[j]=sum*reluDerivative(h2Raw[j]);
                    }

                    for(int j=0;j<h2;j++){
                        dB2[j]+=dH2[j];
                        for(int i1=0;i1<h1;i1++) dW2[i1][j]+=h1v[i1]*dH2[j];
                    }

                    double[] dH1=new double[h1];
                    for(int j=0;j<h1;j++){
                        double sum=0;
                        for(int k=0;k<h2;k++) sum+=w2[j][k]*dH2[k];
                        dH1[j]=sum*reluDerivative(h1Raw[j]);
                    }

                    for(int j=0;j<h1;j++){
                        dB1[j]+=dH1[j];
                        for(int i1=0;i1<inputSize;i1++) dW1[i1][j]+=x[i1]*dH1[j];
                    }
                }

                for(int i=0;i<inputSize;i++) for(int j=0;j<h1;j++) w1[i][j]-=lr*(dW1[i][j]/bs+lambda*w1[i][j]);
                for(int j=0;j<h1;j++) b1[j]-=lr*dB1[j]/bs;
                for(int i=0;i<h1;i++) for(int j=0;j<h2;j++) w2[i][j]-=lr*(dW2[i][j]/bs+lambda*w2[i][j]);
                for(int j=0;j<h2;j++) b2[j]-=lr*dB2[j]/bs;
                for(int i=0;i<h2;i++) for(int j=0;j<outputSize;j++) w3[i][j]-=lr*(dW3[i][j]/bs+lambda*w3[i][j]);
                for(int j=0;j<outputSize;j++) b3[j]-=lr*dB3[j]/bs;
            }

            // Test accuracy
            int correct=0;
            double[] digitCorrect = new double[10];
            double[] digitTotal = new double[10];
            for(int i=0;i<testInputs.size();i++){
                Map<String,Object> f=forward(testInputs.get(i),w1,b1,w2,b2,w3,b3,false);
                double[] out=(double[])f.get("output");
                int pred=0; double max=out[0];
                for(int j=1;j<out.length;j++){ if(out[j]>max){ max=out[j]; pred=j; } }
                int label = testLabels.get(i);
                if(pred==label) correct++;
                digitTotal[label]++;
                if(pred==label) digitCorrect[label]++;
            }
            double acc = correct*1.0/testInputs.size();
            double[] perDigitAcc = new double[10];
            for(int i=0;i<10;i++) perDigitAcc[i]=digitCorrect[i]/digitTotal[i];

            System.out.printf("Epoch %d: Total Accuracy %.2f%%\n",epoch,acc*100);
            visualizer.update(epoch, acc, perDigitAcc);

            // Log to CSV
            csv.write(epoch + "," + acc);
            for(int i=0;i<10;i++) csv.write("," + perDigitAcc[i]);
            csv.write("\n");
            csv.flush();
        }

        csv.close();
        saveWeights("weights.dat", w1,b1,w2,b2,w3,b3);
        System.out.println("Training complete! Weights saved to weights.dat");
    }
}
