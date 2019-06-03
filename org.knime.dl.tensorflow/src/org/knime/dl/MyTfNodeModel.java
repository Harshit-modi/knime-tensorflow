package org.knime.dl;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.knime.core.data.DataCell;
import org.knime.core.data.DataColumnSpec;
import org.knime.core.data.DataColumnSpecCreator;
import org.knime.core.data.DataRow;
import org.knime.core.data.DataTableSpec;
import org.knime.core.data.RowIterator;
import org.knime.core.data.RowKey;
import org.knime.core.data.def.DefaultRow;
import org.knime.core.data.def.DoubleCell;
import org.knime.core.data.def.IntCell;
import org.knime.core.data.def.StringCell;
import org.knime.core.node.BufferedDataContainer;
import org.knime.core.node.BufferedDataTable;
import org.knime.core.node.CanceledExecutionException;
import org.knime.core.node.defaultnodesettings.SettingsModelIntegerBounded;

import org.knime.core.node.ExecutionContext;
import org.knime.core.node.ExecutionMonitor;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeLogger;
import org.knime.core.node.NodeModel;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;

import org.bytedeco.tensorflow.*;
import static org.bytedeco.tensorflow.global.tensorflow.*;

/**
 * This is the model implementation of MyTf.
 * 
 *
 * @author Harshit
 */
public class MyTfNodeModel extends NodeModel {
	
	 /**
     * Checks the status and throws an Exception in case any error occurred
     *
     * @param s
     * @throws Exception
     */
    static void checkStatus(Status s) throws Exception {
        if (!s.ok())
            throw new Exception(s.error_message().getString());
        s.close();
    }
    
	static BufferedDataTable runTensorflow(BufferedDataTable inTable, final ExecutionContext exec) throws Exception {
		// Platform-specific initialization routine
        InitMain("trainer", (int[])null, null);

        // Read the data set
        CarDataset dataset = new CarDataset(inTable);

        // Copy the data into some tensors
        Tensor tensorX = Tensor.create(dataset.flatX(), new TensorShape(dataset.size(), 3));
        Tensor tensorY = Tensor.create(dataset.flatY(), new TensorShape(dataset.size(), 1));

        // Create a new empty graph
        Scope scope = Scope.NewRootScope();

        // Placeholder in the graph where tensors can be loaded into
        Placeholder x = new Placeholder(scope.WithOpName("x"), DT_FLOAT);
        Placeholder y = new Placeholder(scope.WithOpName("y"), DT_FLOAT);

        // Weights initialization
        Variable w1 = new Variable(scope.WithOpName("w1"), new TensorShape(3, 3).asPartialTensorShape(), DT_FLOAT);
        Input rws1 = new Input(Tensor.create(new int[] { 3, 3 }, new TensorShape(new long[] { 2 })));
        RandomNormal rw1 = new RandomNormal(scope, rws1, DT_FLOAT);
        Assign assign_w1 = new Assign(scope.WithOpName("assign_w1"), w1.asInput(), rw1.asInput());

        Variable w2 = new Variable(scope.WithOpName("w2"), new TensorShape(3, 2).asPartialTensorShape(), DT_FLOAT);
        Input rws2 = new Input(Tensor.create(new int[] { 3, 2 }, new TensorShape(new long[] { 2 })));
        RandomNormal rw2 = new RandomNormal(scope, rws2, DT_FLOAT);
        Assign assign_w2 = new Assign(scope.WithOpName("assign_w2"), w2.asInput(), rw2.asInput());

        Variable w3 = new Variable(scope.WithOpName("w3"), new TensorShape(2, 1).asPartialTensorShape(), DT_FLOAT);
        Input rws3 = new Input(Tensor.create(new int[] { 2, 1 }, new TensorShape(new long[] { 2 })));
        RandomNormal rw3 = new RandomNormal(scope, rws3, DT_FLOAT);
        Assign assign_w3 = new Assign(scope.WithOpName("assign_w3"), w3.asInput(), rw3.asInput());

        // Bias initialization
        Variable b1 = new Variable(scope.WithOpName("b1"), new TensorShape(1, 3).asPartialTensorShape(), DT_FLOAT);
        Input rbs1 = new Input(Tensor.create(new int[] { 1, 3 }, new TensorShape(new long[] { 2 })));
        RandomNormal rb1 = new RandomNormal(scope, rbs1, DT_FLOAT);
        Assign assign_b1 = new Assign(scope.WithOpName("assign_b1"), b1.asInput(), rb1.asInput());

        Variable b2 = new Variable(scope.WithOpName("b2"), new TensorShape(1, 2).asPartialTensorShape(), DT_FLOAT);
        Input rbs2 = new Input(Tensor.create(new int[] { 1, 2 }, new TensorShape(new long[] { 2 })));
        RandomNormal rb2 = new RandomNormal(scope, rbs2, DT_FLOAT);
        Assign assign_b2 = new Assign(scope.WithOpName("assign_b2"), b2.asInput(), rb2.asInput());

        Variable b3 = new Variable(scope.WithOpName("b3"), new TensorShape(1, 1).asPartialTensorShape(), DT_FLOAT);
        Input rbs3 = new Input(Tensor.create(new int[] { 1, 1 }, new TensorShape(new long[] { 2 })));
        RandomNormal rb3 = new RandomNormal(scope, rbs3, DT_FLOAT);
        Assign assign_b3 = new Assign(scope.WithOpName("assign_b3"), b3.asInput(), rb3.asInput());

        // Fully connecter layers with an Tanh activation function
        Tanh layer_1 = new Tanh(scope, new Tanh(scope, new Add(scope, new MatMul(scope, x.asInput(), w1.asInput()).asInput(), b1.asInput()).asInput()).asInput());
        Tanh layer_2 = new Tanh(scope, new Add(scope, new MatMul(scope, layer_1.asInput(), w2.asInput()).asInput(), b2.asInput()).asInput());
        Tanh layer_3 = new Tanh(scope, new Add(scope, new MatMul(scope, layer_2.asInput(), w3.asInput()).asInput(), b3.asInput()).asInput());

        // Weight regularization
        L2Loss l1 = new L2Loss(scope, w1.asInput());
        L2Loss l2 = new L2Loss(scope, w2.asInput());
        L2Loss l3 = new L2Loss(scope, w3.asInput());
        AddN regularization = new AddN(scope, new InputList(new OutputVector(l1.asOutput(), l2.asOutput(), l3.asOutput())));

        // Combined loss calculation (prediction loss and weight loss)
        Input axis = new Input(Tensor.create(new int[] { 0, 1 }, new TensorShape(new long[] { 2 })));
        Input scale = new Input(Const(scope, 0.01f));
        Add loss = new Add(scope.WithOpName("loss"),
                new Mean(scope, new Square(scope, new Subtract(scope, layer_3.asInput(), y.asInput()).asInput()).asInput(), axis).asInput(),
                new Multiply(scope, scale, regularization.asInput()).asInput());

        // Add the gradient operations to the graph
        OutputVector node_outputs = new OutputVector(loss.asOutput());
        OutputVector node_inputs = new OutputVector(w1.asOutput(), w2.asOutput(), w3.asOutput(), b1.asOutput(), b2.asOutput(), b3.asOutput());
        OutputVector node_grad_outputs = new OutputVector();
        checkStatus(AddSymbolicGradients(scope, node_outputs, node_inputs, node_grad_outputs));

        // Update the weight and bias values using gradient descent
        Input alpha = new Input(Const(scope.WithOpName("alpha"), 0.01f));
        ApplyGradientDescent apply_w1 = new ApplyGradientDescent(scope.WithOpName("apply_w1"), w1.asInput(), alpha, new Input(node_grad_outputs.get(0)));
        ApplyGradientDescent apply_w2 = new ApplyGradientDescent(scope.WithOpName("apply_w2"), w2.asInput(), alpha, new Input(node_grad_outputs.get(1)));
        ApplyGradientDescent apply_w3 = new ApplyGradientDescent(scope.WithOpName("apply_w3"), w3.asInput(), alpha, new Input(node_grad_outputs.get(2)));
        ApplyGradientDescent apply_b1 = new ApplyGradientDescent(scope.WithOpName("apply_b1"), b1.asInput(), alpha, new Input(node_grad_outputs.get(3)));
        ApplyGradientDescent apply_b2 = new ApplyGradientDescent(scope.WithOpName("apply_b2"), b2.asInput(), alpha, new Input(node_grad_outputs.get(4)));
        ApplyGradientDescent apply_b3 = new ApplyGradientDescent(scope.WithOpName("apply_b3"), b3.asInput(), alpha, new Input(node_grad_outputs.get(5)));

        // Build a graph definition object
        GraphDef def = new GraphDef();
        checkStatus(scope.ToGraphDef(def));

        // Creates a session.
        SessionOptions options = new SessionOptions();
        BufferedDataTable out;
        try(final Session session = new Session(options)) {

            // Create the graph to be used for the session.
            checkStatus(session.Create(def));

            // empty vectors
            StringTensorPairVector input_feed = new StringTensorPairVector();
            StringVector output_tensor_name = new StringVector();
            StringVector target_tensor_name = new StringVector();
            TensorVector outputs = new TensorVector();

            // different
            StringVector target_assign_tensor_name = new StringVector("assign_w1:0", "assign_w2:0", "assign_w3:0", "assign_b1:0", "assign_b2:0", "assign_b3:0");
            StringVector target_apply_tensor_name = new StringVector("apply_w1:0", "apply_w2:0", "apply_w3:0", "apply_b1:0", "apply_b2:0", "apply_b3:0");
            StringTensorPairVector input_xy_feed = new StringTensorPairVector(new String[] {"x", "y"}, new Tensor[] { tensorX, tensorY });
            StringVector output_loss_tensor_name = new StringVector("loss:0");

            // Generate random weights and bias values and assign them
            System.out.println("Setup");
            checkStatus(session.Run(input_feed, output_tensor_name, target_assign_tensor_name, outputs));
            
         // the data table spec of the single output table, 
            // the table will have three columns:
            DataColumnSpec[] allColSpecs = new DataColumnSpec[2];
            allColSpecs[0] = 
                new DataColumnSpecCreator("Column 0", StringCell.TYPE).createSpec();
            allColSpecs[1] = 
                new DataColumnSpecCreator("Column 1", DoubleCell.TYPE).createSpec();
            
            DataTableSpec outputSpec = new DataTableSpec(allColSpecs);
            // the execution context will provide us with storage capacity, in this
            // case a data container to which we will add rows sequentially
            // Note, this container can also handle arbitrary big data tables, it
            // will buffer to disc if necessary.
            BufferedDataContainer container = exec.createDataContainer(outputSpec);
            // let's add m_count rows to it
            
         // Input some training data into the graph
            for (int epoch = 0; epoch < 5000; epoch++) {

                // print loss every 100 epoch
                checkStatus(session.Run(input_xy_feed, output_loss_tensor_name, target_tensor_name, outputs));
                if(epoch % 100 == 0) {
                	RowKey key = new RowKey("Row " + epoch);
                	DataCell[] cells = new DataCell[2];
                	
                    FloatBuffer loss_flat = outputs.get(0).createBuffer();
                    cells[0] = new StringCell("Iteration " + epoch + " with error:");
                    cells[1] = new DoubleCell(loss_flat.get(0));
                    DataRow row = new DefaultRow(key, cells);
                    container.addRowToTable(row);
                    
                    // check if the execution monitor was canceled
                    exec.checkCanceled();
                    exec.setProgress(epoch / 5000, 
                        "Adding row " + epoch);
                }

                // train
                checkStatus(session.Run(input_xy_feed, output_tensor_name, target_apply_tensor_name, outputs));
            }
            // once we are done, we close the container and return its table
            container.close();
            out = container.getTable();
        }

        System.out.println("Finished");
        
        return out;
	}
	
	/**
     * Class containing the data of the normalized_car_features.csv file
     */
    static class CarDataset {

        private final float mean_km, std_km;
        private final float mean_age, std_age;
        private final float min_price, max_price;

        private final List<float[]> x = new ArrayList<>();
        private final List<float[]> y = new ArrayList<>();

        public CarDataset(BufferedDataTable inTable) throws IOException {
            URL url = new URL("https://raw.githubusercontent.com/theflofly/dnn_tensorflow_cpp/master/normalized_car_features.csv");
            try(InputStream is = url.openStream()) {
                try(BufferedReader br = new BufferedReader(new InputStreamReader(is, "ASCII"))) {

                    // first row
                	
                	String temp = "104272.93201133145,65391.707582584619,6.0168838526912181,3.40778120276032,1500.0,124000.0";
                    String[] meta_columns = temp.split(",");
                    mean_km = Float.parseFloat(meta_columns[0]);
                    std_km = Float.parseFloat(meta_columns[1]);
                    mean_age = Float.parseFloat(meta_columns[2]);
                    std_age = Float.parseFloat(meta_columns[3]);
                    min_price = Float.parseFloat(meta_columns[4]);
                    max_price = Float.parseFloat(meta_columns[5]);

                    // ignore second row
                    br.readLine();

                    // read the remaining lines
                    
                    for (RowIterator it = inTable.iterator(); it.hasNext();) {
                        DataRow row = it.next();
                        
                        List<String> colList = row.stream().map(x->x.toString()).collect(Collectors.toList());
                        
                        String[] cols = colList.toArray(new String[colList.size()]);
                        x.add(new float[] { Float.parseFloat(cols[0]), Float.parseFloat(cols[1]), Float.parseFloat(cols[2]) });
                        y.add(new float[] { Float.parseFloat(cols[3]) });
                        
                	}
                }
            }
        }

        @Override
        public String toString() {
            return x.size() + " data points with min/max price of "+min_price+"/"+max_price+
                    ", mean/std age of "+mean_age+"/"+std_age+" and km "+mean_km+"/"+std_km;
        }

        /**
         * Number of entries in the data set
         *
         * @return
         */
        public int size() {
            return y.size();
        }

        /**
         * All x values in scan line order
         *
         * @return
         */
        public float[] flatX() {
            return flatten(x);
        }

        /**
         * All y values in scan line order
         *
         * @return
         */
        public float[] flatY() {
            return flatten(y);
        }

        /**
         * Returns all values of the input in scan line order
         *
         * @param input
         * @return
         */
        private static float[] flatten(List<float[]> input) {

            // sum of all input array sizes
            int size = 0;
            for (float[] datapoint : input)
                size += datapoint.length;

            // copy data into a new single 1d array
            int i = 0;
            float[] result = new float[size];
            for (float[] datapoint : input)
                for (float value : datapoint)
                    result[i++] = value;

            return result;
        }
    }

	
    // the logger instance
    private static final NodeLogger logger = NodeLogger
            .getLogger(MyTfNodeModel.class);
        
    /** the settings key which is used to retrieve and 
        store the settings (from the dialog or from a settings file)    
       (package visibility to be usable from the dialog). */
	static final String CFGKEY_COUNT = "Count";

    /** initial default count value. */
    static final int DEFAULT_COUNT = 100;

    // example value: the models count variable filled from the dialog 
    // and used in the models execution method. The default components of the
    // dialog work with "SettingsModels".
    private final SettingsModelIntegerBounded m_count =
        new SettingsModelIntegerBounded(MyTfNodeModel.CFGKEY_COUNT,
                    MyTfNodeModel.DEFAULT_COUNT,
                    Integer.MIN_VALUE, Integer.MAX_VALUE);
    

    /**
     * Constructor for the node model.
     */
    protected MyTfNodeModel() {
    
        // TODO one incoming port and one outgoing port is assumed
        super(1, 1);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected BufferedDataTable[] execute(final BufferedDataTable[] inData,
            final ExecutionContext exec) throws Exception {

        // TODO do something here
        logger.info("Node Model Stub... this is not yet implemented !");

        BufferedDataTable out = runTensorflow(inData[0], exec);
        
//        // the data table spec of the single output table, 
//        // the table will have three columns:
//        DataColumnSpec[] allColSpecs = new DataColumnSpec[3];
//        allColSpecs[0] = 
//            new DataColumnSpecCreator("Column 0", StringCell.TYPE).createSpec();
//        allColSpecs[1] = 
//            new DataColumnSpecCreator("Column 1", DoubleCell.TYPE).createSpec();
//        allColSpecs[2] = 
//            new DataColumnSpecCreator("Column 2", IntCell.TYPE).createSpec();
//        DataTableSpec outputSpec = new DataTableSpec(allColSpecs);
//        // the execution context will provide us with storage capacity, in this
//        // case a data container to which we will add rows sequentially
//        // Note, this container can also handle arbitrary big data tables, it
//        // will buffer to disc if necessary.
//        BufferedDataContainer container = exec.createDataContainer(outputSpec);
//        // let's add m_count rows to it
//        for (int i = 0; i < m_count.getIntValue(); i++) {
//            RowKey key = new RowKey("Row " + i);
//            // the cells of the current row, the types of the cells must match
//            // the column spec (see above)
//            DataCell[] cells = new DataCell[3];
//            cells[0] = new StringCell("String_" + "_" + i); 
//            cells[1] = new DoubleCell(0.5 * i); 
//            cells[2] = new IntCell(i);
//            DataRow row = new DefaultRow(key, cells);
//            container.addRowToTable(row);
//            
//            // check if the execution monitor was canceled
//            exec.checkCanceled();
//            exec.setProgress(i / (double)m_count.getIntValue(), 
//                "Adding row " + i);
//        }
//        // once we are done, we close the container and return its table
//        container.close();
//        BufferedDataTable out = container.getTable();
        return new BufferedDataTable[]{out};
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void reset() {
        // TODO Code executed on reset.
        // Models build during execute are cleared here.
        // Also data handled in load/saveInternals will be erased here.
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected DataTableSpec[] configure(final DataTableSpec[] inSpecs)
            throws InvalidSettingsException {
        
        // TODO: check if user settings are available, fit to the incoming
        // table structure, and the incoming types are feasible for the node
        // to execute. If the node can execute in its current state return
        // the spec of its output data table(s) (if you can, otherwise an array
        // with null elements), or throw an exception with a useful user message

        return new DataTableSpec[]{null};
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void saveSettingsTo(final NodeSettingsWO settings) {

        // TODO save user settings to the config object.
        
        m_count.saveSettingsTo(settings);

    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void loadValidatedSettingsFrom(final NodeSettingsRO settings)
            throws InvalidSettingsException {
            
        // TODO load (valid) settings from the config object.
        // It can be safely assumed that the settings are valided by the 
        // method below.
        
        m_count.loadSettingsFrom(settings);

    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void validateSettings(final NodeSettingsRO settings)
            throws InvalidSettingsException {
            
        // TODO check if the settings could be applied to our model
        // e.g. if the count is in a certain range (which is ensured by the
        // SettingsModel).
        // Do not actually set any values of any member variables.

        m_count.validateSettings(settings);

    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    protected void loadInternals(final File internDir,
            final ExecutionMonitor exec) throws IOException,
            CanceledExecutionException {
        
        // TODO load internal data. 
        // Everything handed to output ports is loaded automatically (data
        // returned by the execute method, models loaded in loadModelContent,
        // and user settings set through loadSettingsFrom - is all taken care 
        // of). Load here only the other internals that need to be restored
        // (e.g. data used by the views).

    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    protected void saveInternals(final File internDir,
            final ExecutionMonitor exec) throws IOException,
            CanceledExecutionException {
       
        // TODO save internal models. 
        // Everything written to output ports is saved automatically (data
        // returned by the execute method, models saved in the saveModelContent,
        // and user settings saved through saveSettingsTo - is all taken care 
        // of). Save here only the other internals that need to be preserved
        // (e.g. data used by the views).

    }

}

