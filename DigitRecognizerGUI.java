// DigitRecognizerGUI.java
import javax.swing.*;
import javax.swing.BorderFactory;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;

public class DigitRecognizerGUI extends JFrame {
    private BufferedImage highResCanvas;
    private int brushSize = 20;

    // CNN + Dense weights (matching DigitRecognizer.saveWeightsObj / loadWeightsObj)
    private DigitRecognizer.ConvBlockAdvanced conv;
    private double[][] W1, W2, W3;
    private double[] b1, b2, b3;

    private JLabel predictionLabel;

    // adjust if your trainer saved a different filename
    private static final String WEIGHTS_FILE = "weights.obj";

    public DigitRecognizerGUI() {
        super("Digit Recognizer");
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setLayout(new BorderLayout());

        highResCanvas = new BufferedImage(420, 420, BufferedImage.TYPE_BYTE_GRAY);
        clearCanvas();

        JPanel drawPanel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                g.drawImage(highResCanvas, 0, 0, null);
            }
        };
        drawPanel.setPreferredSize(new Dimension(420, 420));
        drawPanel.setBackground(Color.BLACK);

        MouseAdapter drawAdapter = new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                drawAt(e.getX(), e.getY());
                drawPanel.repaint();
                updatePredictions();
            }
            @Override
            public void mouseDragged(MouseEvent e) {
                drawAt(e.getX(), e.getY());
                drawPanel.repaint();
                updatePredictions();
            }
        };
        drawPanel.addMouseListener(drawAdapter);
        drawPanel.addMouseMotionListener(drawAdapter);

        // Prediction display (scrollable)
        predictionLabel = new JLabel("<html>All Predictions (most to least probable):<br><br></html>");
        predictionLabel.setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));
        JScrollPane scroll = new JScrollPane(predictionLabel);
        scroll.setPreferredSize(new Dimension(240, 420));

        // Layout
        JPanel mainPanel = new JPanel(new BorderLayout());
        mainPanel.add(drawPanel, BorderLayout.CENTER);
        mainPanel.add(scroll, BorderLayout.EAST);
        add(mainPanel, BorderLayout.CENTER);

        // Keyboard: clear with 'c'
        drawPanel.setFocusable(true);
        drawPanel.requestFocus();
        drawPanel.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                if (e.getKeyChar() == 'c' || e.getKeyChar() == 'C') {
                    clearCanvas();
                    drawPanel.repaint();
                    updatePredictions();
                }
            }
        });

        // Load weights produced by the trainer (weights.obj)
        try {
            Object[] weights = DigitRecognizer.loadWeightsObj(WEIGHTS_FILE);
            // Order: K1, B1, K2, B2, W1, b1, W2, b2, W3, b3
            double[][][] K1 = (double[][][]) weights[0];
            double[]  B1  = (double[])       weights[1];
            double[][][] K2 = (double[][][]) weights[2];
            double[]  B2  = (double[])       weights[3];
            W1 = (double[][]) weights[4]; b1 = (double[]) weights[5];
            W2 = (double[][]) weights[6]; b2 = (double[]) weights[7];
            W3 = (double[][]) weights[8]; b3 = (double[]) weights[9];

            // Build a ConvBlockAdvanced with correct numbers of filters and inject weights
            int numF1 = K1.length;
            int numF2 = K2.length;
            conv = new DigitRecognizer.ConvBlockAdvanced(numF1, numF2, new Random(0));

            // Assign loaded kernels/biases into conv instance
            // conv.K1, conv.B1, conv.K2, conv.B2 are package-visible in DigitRecognizer code
            conv.K1 = K1;
            conv.B1 = B1;
            conv.K2 = K2;
            conv.B2 = B2;

        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this,
                "Failed to load weights file '" + WEIGHTS_FILE + "'.\nMake sure the trainer saved that file in the working directory.\n\n" + ex.getMessage(),
                "Error",
                JOptionPane.ERROR_MESSAGE);
            ex.printStackTrace();
            System.exit(1);
        }

        pack();
        setResizable(false);
        setLocationRelativeTo(null);
    }

    private void drawAt(int x, int y) {
        Graphics2D g = highResCanvas.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setColor(Color.WHITE);
        g.fillOval(x - brushSize / 2, y - brushSize / 2, brushSize, brushSize);
        g.dispose();
    }

    private void clearCanvas() {
        Graphics2D g = highResCanvas.createGraphics();
        g.setColor(Color.BLACK);
        g.fillRect(0, 0, highResCanvas.getWidth(), highResCanvas.getHeight());
        g.dispose();
    }

    private void updatePredictions() {
        double[] input = getInputFromCanvas(); // 28x28 normalized to [0,1]

        // --- CNN forward (no dropout) ---
        double[] convFeat = conv.forward(input);
        DigitRecognizer.DenseCache cache = DigitRecognizer.denseForward(
                convFeat, W1, b1, W2, b2, W3, b3,
                false,     // training = false
                0.0,       // dropoutRate = 0
                new Random(0));

        double[] probs = cache.out; // already softmaxed

        Integer[] idx = new Integer[probs.length];
        for (int i = 0; i < probs.length; i++) idx[i] = i;
        Arrays.sort(idx, Comparator.comparingDouble(i -> -probs[i]));

        StringBuilder sb = new StringBuilder("<html>All Predictions (most to least probable):<br><br>");
        for (int rank = 0; rank < probs.length; rank++) {
            int d = idx[rank];
            String style = rank == 0 ? "font-weight: bold;" : "";
            sb.append("<span style='").append(style).append("'>")
              .append(d).append(" → ")
              .append(String.format("%.2f", probs[d]*100)).append("%</span><br>");
        }
        sb.append("</html>");

        SwingUtilities.invokeLater(() -> predictionLabel.setText(sb.toString()));
    }

    private double[] getInputFromCanvas() {
        // Downscale canvas -> 28x28, find bounding box, scale longest side to 20px, center in 28x28
        BufferedImage small = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = small.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(highResCanvas, 0, 0, 28, 28, null);
        g.dispose();

        int minX = 28, minY = 28, maxX = -1, maxY = -1;
        for (int y = 0; y < 28; y++) for (int x = 0; x < 28; x++) {
            int gray = small.getRGB(x, y) & 0xFF;
            if (gray > 10) { // foreground threshold
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }

        if (maxX < 0 || maxY < 0) return new double[28*28]; // empty

        int width = maxX - minX + 1;
        int height = maxY - minY + 1;

        int newW = width > height ? 20 : (int)Math.round((width*20.0)/height);
        int newH = width > height ? (int)Math.round((height*20.0)/width) : 20;

        if (newW < 1) newW = 1;
        if (newH < 1) newH = 1;

        BufferedImage cropped = small.getSubimage(minX, minY, width, height);
        BufferedImage scaled = new BufferedImage(newW, newH, BufferedImage.TYPE_BYTE_GRAY);
        g = scaled.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(cropped, 0, 0, newW, newH, null);
        g.dispose();

        BufferedImage centered = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        g = centered.createGraphics();
        g.setColor(Color.BLACK);
        g.fillRect(0, 0, 28, 28);
        int xOffset = (28 - newW)/2;
        int yOffset = (28 - newH)/2;
        g.drawImage(scaled, xOffset, yOffset, null);
        g.dispose();

        double[] input = new double[28*28];
        for (int y = 0; y < 28; y++) for (int x = 0; x < 28; x++) {
            int gray = centered.getRGB(x, y) & 0xFF;
            input[y*28 + x] = gray / 255.0;
        }
        return input;
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new DigitRecognizerGUI().setVisible(true));
    }
}
