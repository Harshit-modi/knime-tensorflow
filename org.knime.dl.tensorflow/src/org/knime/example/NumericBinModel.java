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

import java.util.ArrayList;
import java.util.List;

import org.knime.core.node.ModelContentWO;

/**
 * Represents the external model of the NumericBinner Node.
 * Stores the lower and upper bound for each bin.
 * 
 * @author KNIME.com AG, Zurich, Switzerland
 */
public class NumericBinModel {
    
    private static final String INTERVAL = "interval";
    private static final String LOWER_BOUND = "lowerBound";
    private static final String UPPER_BOUND = "upperBound";
    private static final String NUMBER_OF_BINS = "numberOfBins";
    
    
    private List<Interval> m_intervals;
    
    /**
     * A NumericBin model represents the intervals making up the bins. 
     *
     */
    public NumericBinModel() {
        m_intervals = new ArrayList<Interval>();
    }

    /**
     * 
     * @param binNumber the number of the bin for which the lower bound 
     * should be returned
     * @return the lower bound of the specified interval.
     */
    public double getLowerBoundForInterval(final int binNumber) {
        return m_intervals.get(binNumber).getLowerBound();
    }

    /**
     * 
     * @param binNumber the number of the bin for which the upper bound 
     * should be returned
     * @return the upper bound of the specified interval.
     */
    public double getUpperBoundForInterval(final int binNumber) {
        return m_intervals.get(binNumber).getUpperBound();
    }
    
    /**
     * 
     * @return the number of bins, i.e. intervals.
     */
    public int getNumberOfBins() {
        return m_intervals.size();
    }
    

    /**
     * Adds an interval to this model.
     * @param lowerBound the lower bound of the interval
     * @param upperBound the upper bound of the interval
     */
    public void addInterval(final double lowerBound, final double upperBound) {
        Interval interval = new Interval(lowerBound, upperBound);
        m_intervals.add(interval);
    }
    
    
    /**
     * Saves this model to the model content.
     * @param modelContent the model content to save to
     */
    public void saveTo(final ModelContentWO modelContent) {
        modelContent.addInt(NUMBER_OF_BINS, m_intervals.size());
        int intervalNr = 0;
        for (Interval interval : m_intervals) {
            ModelContentWO intervalModel = modelContent.addModelContent(
                    INTERVAL + intervalNr++);
            intervalModel.addDouble(LOWER_BOUND, interval.getLowerBound());
            intervalModel.addDouble(UPPER_BOUND, interval.getUpperBound());
        }
    }
    
    /**
     * This class represents an interval of a bin.
     * 
     * @author Fabian Dill, University of Konstanz
     */
    public final class Interval {
        
        private final double m_lowerBound;
        private final double m_upperBound;
        
        /** 
         * Creats an interval with a lower and an upper bound.
         * @param lowerBound the lower bound of the interval
         * @param upperBound the upper bound of the interval
         */
        public Interval(final double lowerBound, final double upperBound) {
            m_lowerBound = lowerBound;
            m_upperBound = upperBound;
        }
        
        /**
         * 
         * @return the lower bound of the interval
         */
        public double getLowerBound() {
            return m_lowerBound;
        }
        
        /**
         * 
         * @return the upper ound of the interval
         */
        public double getUpperBound() {
            return m_upperBound;
        }
    }
}

