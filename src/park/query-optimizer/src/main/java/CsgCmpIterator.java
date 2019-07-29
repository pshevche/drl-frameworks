import java.util.*;
import org.apache.calcite.util.ImmutableBitSet;

public interface CsgCmpIterator {
    public Iterator<ImmutableBitSet[]> csgCmpIterator();
}

