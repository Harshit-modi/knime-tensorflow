package org.knime.dl;

import org.knime.core.node.NodeDialogPane;
import org.knime.core.node.NodeFactory;
import org.knime.core.node.NodeView;

/**
 * <code>NodeFactory</code> for the "MyTf" Node.
 * 
 *
 * @author Harshit
 */
public class MyTfNodeFactory 
        extends NodeFactory<MyTfNodeModel> {

    /**
     * {@inheritDoc}
     */
    @Override
    public MyTfNodeModel createNodeModel() {
        return new MyTfNodeModel();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int getNrNodeViews() {
        return 1;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public NodeView<MyTfNodeModel> createNodeView(final int viewIndex,
            final MyTfNodeModel nodeModel) {
        return new MyTfNodeView(nodeModel);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean hasDialog() {
        return true;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public NodeDialogPane createNodeDialogPane() {
        return new MyTfNodeDialog();
    }

}

