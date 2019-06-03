/*
 * ------------------------------------------------------------------------
  * This source code, its documentation and all appendant files
 * are protected by copyright law. All rights reserved.
 *
 * Copyright, 2008 - 2012
 * KNIME.com, Zurich, Switzerland
 *
 * You may not modify, publish, transmit, transfer or sell, reproduce,
 * create derivative works from, distribute, perform, display, or in
 * any way exploit any of the content, in whole or in part, except as
 * otherwise expressly permitted in writing by the copyright owner or
 * as specified in the license file distributed with this product.
 *
 * If you have any questions please contact the copyright holder:
 * website: www.knime.com
 * email: contact@knime.com
 * ---------------------------------------------------------------------
 */
package org.knime.example;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.knime.core.data.DataCell;
import org.knime.core.data.DataColumnDomain;
import org.knime.core.data.DataColumnDomainCreator;
import org.knime.core.data.DataColumnSpec;
import org.knime.core.data.DataColumnSpecCreator;
import org.knime.core.data.DataRow;
import org.knime.core.data.DataTableSpec;
import org.knime.core.data.DoubleValue;
import org.knime.core.data.RowIterator;
import org.knime.core.data.container.CellFactory;
import org.knime.core.data.container.ColumnRearranger;
import org.knime.core.data.def.DoubleCell;
import org.knime.core.data.def.IntCell;
import org.knime.core.node.BufferedDataTable;
import org.knime.core.node.CanceledExecutionException;
import org.knime.core.node.ExecutionContext;
import org.knime.core.node.ExecutionMonitor;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.ModelContent;
import org.knime.core.node.ModelContentRO;
import org.knime.core.node.ModelContentWO;
import org.knime.core.node.NodeModel;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.knime.core.node.defaultnodesettings.SettingsModelIntegerBounded;
import org.knime.core.node.defaultnodesettings.SettingsModelString;

/**
 * This is the model implementation of NumericBinner.
 * Simple numeric binner with equidistant bins
 *
 * @author KNIME.com AG, Zurich, Switzerland
 */
public class NumericBinnerNodeModel extends NodeModel {

    /** Constant for the inport index. */
    public static final int IN_PORT = 0;
    
    // ************ fields for the settings ***************
    
    /** The config key for the number of bins. */ 
    public static final String CFGKEY_NR_OF_BINS = "numberOfbins"; 
    /** The config key for the selected column. */
    public static final String CFGKEY_COLUMN_NAME = "columnName";

    /** Default number of bins. */
    public static final int DEFAULT_NR_OF_BINS = 10;


    // the settings model for the number of bins 
    private final SettingsModelIntegerBounded m_numberOfBins =
        new SettingsModelIntegerBounded(
                 NumericBinnerNodeModel.CFGKEY_NR_OF_BINS,
                 NumericBinnerNodeModel.DEFAULT_NR_OF_BINS,
                 1, Integer.MAX_VALUE);
    
    // the settings model storing the column to bin
    private final SettingsModelString m_column = new SettingsModelString(
            NumericBinnerNodeModel.CFGKEY_COLUMN_NAME, "");


    // ************* fields needed for execution **************
    
    private NumericBin[] m_bins;
    
    // *********** Internal Model Keys:*************
    
    private static final String FILE_NAME = "numericBinnerInternals.xml";
    
    private static final String INTERNAL_MODEL = "internalModel";
    
    private static final String NUMERIC_BIN = "numericBin";
    
    // the external model
    private NumericBinModel m_model;    
    
    /**
     * Constructor for the node model with one in-port for the data to bin
     * and one out-port for the input data with an additional column with 
     * the binning information.
     */
    protected NumericBinnerNodeModel() {
        super(1, 1);
    }

    /** {@inheritDoc} */
    @Override
    protected BufferedDataTable[] execute(final BufferedDataTable[] inData,
            final ExecutionContext exec) throws Exception {
    	
    	BufferedDataTable inputDataTable = inData[0];
    	
    	for (RowIterator it = inputDataTable.iterator(); it.hasNext();) {
            DataRow row = it.next();
            
            List<String> cols = row.stream().map(x->x.toString()).collect(Collectors.toList());
            
            for(String s: cols) {
            	System.out.println(s);
            }
    	}
        // first we get the column spec of the selected column
        DataColumnSpec colSpec = inData[IN_PORT].getDataTableSpec()
            .getColumnSpec(m_column.getStringValue());
        double lowerBound = 0;
        double upperBound = 0;
        DataColumnDomain domain = colSpec.getDomain();
        // check if we have to calculate the lower and upper bound
        if (domain == null 
                || !domain.hasBounds()) {
            domain = calculateDomainValues(inData[IN_PORT]);
        }
        lowerBound = ((DoubleValue)domain.getLowerBound()).getDoubleValue();
        upperBound = ((DoubleValue)domain.getUpperBound()).getDoubleValue();
        
        // we also need the data structure for the bins
        m_bins = new NumericBin[m_numberOfBins.getIntValue()];
        double interval = (upperBound - lowerBound) 
                / (m_numberOfBins.getIntValue());
        List<Double>splitPoints = new ArrayList<Double>(
                m_numberOfBins.getIntValue());
        double intervalUpperBound = lowerBound;
        // create the external model
        m_model = new NumericBinModel();
        double intervalLowerBound = lowerBound;
        for (int i = 0; i < m_numberOfBins.getIntValue(); i++) {
            intervalLowerBound = intervalUpperBound;
            intervalUpperBound += interval;
            // fill the external model
            m_model.addInterval(intervalLowerBound, intervalUpperBound);
            splitPoints.add(intervalUpperBound);
            // fill the bins with empty representations
            m_bins[i] = new NumericBin();
        }
        
        // now go through the data and bin it first get the column index
        int colIndex = inData[IN_PORT].getDataTableSpec()
            .findColumnIndex(m_column.getStringValue());
        
 
        // instantiate the cell factory
        CellFactory cellFactory = new NumericBinnerCellFactory(
                createOutputColumnSpec(), splitPoints, colIndex, m_bins);
        // create the column rearranger
        ColumnRearranger outputTable = new ColumnRearranger(
                inData[IN_PORT].getDataTableSpec());
        // append the new column
        outputTable.append(cellFactory);
        // and create the actual output table
        BufferedDataTable bufferedOutput = exec.createColumnRearrangeTable(
                inData[IN_PORT], outputTable, exec);
        // return it
        return new BufferedDataTable[]{bufferedOutput};
    }
       
    private DataColumnDomain calculateDomainValues(
            final BufferedDataTable input) {
        int colIndex = input.getDataTableSpec().findColumnIndex(
                m_column.getStringValue());
        double lowerBound = Double.MAX_VALUE;
        double upperBound = Double.MIN_VALUE;
        for (DataRow currRow : input) {
            DataCell currCell = currRow.getCell(colIndex);
            double currValue = ((DoubleValue)currCell).getDoubleValue();
            lowerBound = Math.min(lowerBound, currValue);
            upperBound = Math.max(upperBound, currValue);
        }
        DataColumnDomainCreator domainCreator = new DataColumnDomainCreator();
        domainCreator.setLowerBound(new DoubleCell(lowerBound));
        domainCreator.setUpperBound(new DoubleCell(upperBound));
        return domainCreator.createDomain();
    }  
        
    /**
     * @return the representation of the bins.
     */
    public NumericBin[] getBinRepresentations() {
        return m_bins;
    }    

    /** {@inheritDoc} */
    @Override
    protected void reset() {
        m_bins = null;
    }

    /** {@inheritDoc} */
    @Override
    protected DataTableSpec[] configure(final DataTableSpec[] inSpecs)
            throws InvalidSettingsException {
        // first of all validate the incoming data table spec
        boolean hasNumericColumn = false;
        boolean containsName = false;
        for (int i = 0; i < inSpecs[IN_PORT].getNumColumns(); i++) {
            DataColumnSpec columnSpec = inSpecs[IN_PORT].getColumnSpec(i);
            // we can only work with it, if it contains at least one 
            // numeric column
            if (columnSpec.getType().isCompatible(DoubleValue.class)) {
                // found one numeric column
                hasNumericColumn = true;
            }
            // and if the column name is set it must be contained in the data 
            // table spec
            if (m_column != null 
                    && columnSpec.getName().equals(m_column.getStringValue())) {
                containsName = true;
            }
            
        }
        if (!hasNumericColumn) {
            throw new InvalidSettingsException("Input table must contain at " 
                    + "least one numeric column");
        }
        
        if (!containsName) {
            throw new InvalidSettingsException("Input table contains not the " 
                    + "column " + m_column.getStringValue() + ". Please "
                    + "(re-)configure the node.");
        }
        
        // so far the input is checked and the algorithm can work with the 
        // incoming data
        
        // now produce the output table spec,  
        // i.e. specify the output of this node
        DataColumnSpec newColumnSpec = createOutputColumnSpec();
        // and the DataTableSpec for the appended part
        DataTableSpec appendedSpec = new DataTableSpec(newColumnSpec);
        // since it is only appended the new output spec contains both:
        // the original spec and the appended one
        DataTableSpec outputSpec = new DataTableSpec(
                inSpecs[IN_PORT], appendedSpec);
        return new DataTableSpec[]{outputSpec};
    } 
    
    private DataColumnSpec createOutputColumnSpec() {
        // we want to add a column with the number of the bin 
        DataColumnSpecCreator colSpecCreator = new DataColumnSpecCreator(
                "Bin Number", IntCell.TYPE);
        // if we know the number of bins we also know the number of possible
        // values of that new column
        DataColumnDomainCreator domainCreator = new DataColumnDomainCreator(
                new IntCell(0), new IntCell(m_numberOfBins.getIntValue() - 1));
        // and can add this domain information to the output spec
        colSpecCreator.setDomain(domainCreator.createDomain());
        // now the column spec can be created
        DataColumnSpec newColumnSpec = colSpecCreator.createSpec();
        return newColumnSpec;
    }    

    /** {@inheritDoc} */
    @Override
    protected void saveSettingsTo(final NodeSettingsWO settings) {
        // save settings to the config object.
        m_numberOfBins.saveSettingsTo(settings);
        m_column.saveSettingsTo(settings);
    }

    /** {@inheritDoc} */
    @Override
    protected void loadValidatedSettingsFrom(final NodeSettingsRO settings)
            throws InvalidSettingsException {
        // loads the values from the settings into the models. It can be safely 
        // assumed that the settings are validated by the method below
        m_numberOfBins.loadSettingsFrom(settings);
        m_column.loadSettingsFrom(settings);

    }

    /** {@inheritDoc} */
    @Override
    protected void validateSettings(final NodeSettingsRO settings)
            throws InvalidSettingsException {
        // delegate this to the settings models
        m_numberOfBins.validateSettings(settings);
        m_column.validateSettings(settings);
    }
    
    /** {@inheritDoc} */
    @Override
    protected void loadInternals(final File internDir,
            final ExecutionMonitor exec) throws IOException,
            CanceledExecutionException {
        m_bins = new NumericBin[m_numberOfBins.getIntValue()];
        File file = new File(internDir, FILE_NAME);
        FileInputStream fis = new FileInputStream(file);
        ModelContentRO modelContent = ModelContent.loadFromXML(fis);
        try {
            for (int i = 0; i < m_numberOfBins.getIntValue(); i++) {
                NumericBin bin = new NumericBin();
                ModelContentRO subModelContent = modelContent
                        .getModelContent(NUMERIC_BIN + i);
                bin.loadFrom(subModelContent);
                m_bins[i] = bin;
            }
        } catch (InvalidSettingsException e) {
            throw new IOException(e.getMessage());
        }
    }
    
    /** {@inheritDoc} */
    @Override
    protected void saveInternals(final File internDir,
            final ExecutionMonitor exec) throws IOException,
            CanceledExecutionException {
        if (m_bins != null) {
            // create the main model content
            ModelContent modelContent = new ModelContent(INTERNAL_MODEL);
            for (int i = 0; i < m_bins.length; i++) {
                // for each bin create a sub model content
                ModelContentWO subContent = modelContent
                        .addModelContent(NUMERIC_BIN + i);
                // save the bin to the sub model content
                m_bins[i].saveTo(subContent);
            }
            // now all bins are stored to the model content
            // but the model content must be written to XML
            // internDir is the directory for this node
            File file = new File(internDir, FILE_NAME);
            FileOutputStream fos = new FileOutputStream(file);
            modelContent.saveToXML(fos);
        }
    }

}
