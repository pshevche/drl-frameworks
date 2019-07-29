import org.apache.calcite.plan.RelOptCost;
import org.apache.calcite.plan.RelOptCostFactory;
import org.apache.calcite.plan.RelOptUtil;

import java.util.Objects;

/**
 * <code>MyCost</code> represents the cost of a plan node.
 *
 * <p>This class is immutable: none of the methods modify any member
 * variables.</p>
 */

// FIXME: change all comparison operators to consider cost instead of these
// other things.
class MyCost implements RelOptCost {
  //~ Static fields/initializers ---------------------------------------------

  static final MyCost INFINITY =
      new MyCost(
          Double.POSITIVE_INFINITY,
          Double.POSITIVE_INFINITY,
          Double.POSITIVE_INFINITY) {
        public String toString() {
          return "{inf}";
        }
      };

  static final MyCost HUGE =
      new MyCost(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE) {
        public String toString() {
          return "{huge}";
        }
      };

  static final MyCost ZERO =
      new MyCost(0.0, 0.0, 0.0) {
        public String toString() {
          return "{0}";
        }
      };

  static final MyCost TINY =
      new MyCost(1.0, 1.0, 0.0) {
        public String toString() {
          return "{tiny}";
        }
      };

  public static final RelOptCostFactory FACTORY = new Factory();

  //~ Instance fields --------------------------------------------------------

  final double cpu;
  final double io;
  double rowCount;
  // use rowCount, and other factors, to determine an actual cost.
  // Currently, will just be set by the MetadataProvider creating this.
  // FIXME: better way to integrate this?
  public double cost;

  //~ Constructors -----------------------------------------------------------

  MyCost(double rowCount, double cpu, double io) {
    this.rowCount = rowCount;
    this.cpu = cpu;
    this.io = io;
    this.cost = rowCount;
  }

  MyCost(double rowCount, double cpu, double io, double cost) {
    this.rowCount = rowCount;
    this.cpu = cpu;
    this.io = io;
    this.cost = cost;
  }

  //~ Methods ----------------------------------------------------------------

  public double getCpu() {
    return cpu;
  }

  public boolean isInfinite() {
    return (this == INFINITY)
        || (this.rowCount == Double.POSITIVE_INFINITY)
        || (this.cpu == Double.POSITIVE_INFINITY)
        || (this.io == Double.POSITIVE_INFINITY);
  }

  public double getIo() {
    return io;
  }

  public boolean isLe(RelOptCost other) {
		MyCost that = (MyCost) other;
    return this.cost <= that.cost;

		//if (true) {
			//return this == that
          //|| this.rowCount <= that.rowCount;
		//}
		//return (this == that)
				//|| ((this.rowCount <= that.rowCount)
				//&& (this.cpu <= that.cpu)
				//&& (this.io <= that.io));
  }

  public boolean isLt(RelOptCost other) {
    if (true) {
      MyCost that = (MyCost) other;
      return this.cost < that.cost;
    }
    return isLe(other) && !equals(other);
  }

  public double getRows() {
    return rowCount;
  }

  public double getCost() {
    return cost;
  }

  @Override public int hashCode() {
    return Objects.hash(rowCount, cpu, io, cost);
  }

  public boolean equals(RelOptCost other) {
    return this == other
        || other instanceof MyCost
        && (this.rowCount == ((MyCost) other).rowCount)
        && (this.cpu == ((MyCost) other).cpu)
        && (this.io == ((MyCost) other).io)
        && (this.cost == ((MyCost) other).cost);
  }

  public boolean isEqWithEpsilon(RelOptCost other) {
    if (!(other instanceof MyCost)) {
      return false;
    }
    MyCost that = (MyCost) other;
    return (this == that)
        || ((Math.abs(this.rowCount - that.rowCount) < RelOptUtil.EPSILON)
        && (Math.abs(this.cpu - that.cpu) < RelOptUtil.EPSILON)
        && (Math.abs(this.io - that.io) < RelOptUtil.EPSILON)
        && (Math.abs(this.cost - that.cost) < RelOptUtil.EPSILON));
  }

  public RelOptCost minus(RelOptCost other) {
    if (this == INFINITY) {
      return this;
    }
    MyCost that = (MyCost) other;
    return new MyCost(
        this.rowCount - that.rowCount,
        this.cpu - that.cpu,
        this.io - that.io,
        this.cost - that.cost);
  }

  public RelOptCost multiplyBy(double factor) {
    if (this == INFINITY) {
      return this;
    }
    return new MyCost(rowCount * factor, cpu * factor, io * factor, cost * factor);
  }

  // FIXME: update this.
  public double divideBy(RelOptCost cost) {
    // Compute the geometric average of the ratios of all of the factors
    // which are non-zero and finite.
    MyCost that = (MyCost) cost;
    double d = 1;
    double n = 0;
    if ((this.rowCount != 0)
        && !Double.isInfinite(this.rowCount)
        && (that.rowCount != 0)
        && !Double.isInfinite(that.rowCount)) {
      d *= this.rowCount / that.rowCount;
      ++n;
    }
    if ((this.cpu != 0)
        && !Double.isInfinite(this.cpu)
        && (that.cpu != 0)
        && !Double.isInfinite(that.cpu)) {
      d *= this.cpu / that.cpu;
      ++n;
    }
    if ((this.io != 0)
        && !Double.isInfinite(this.io)
        && (that.io != 0)
        && !Double.isInfinite(that.io)) {
      d *= this.io / that.io;
      ++n;
    }
    if (n == 0) {
      return 1.0;
    }
    return Math.pow(d, 1 / n);
  }

  public RelOptCost plus(RelOptCost other) {
    MyCost that = (MyCost) other;
    if ((this == INFINITY) || (that == INFINITY)) {
      return INFINITY;
    }
    return new MyCost(
        this.rowCount + that.rowCount,
        this.cpu + that.cpu,
        this.io + that.io,
        this.cost + that.cost);
  }

  public String toString() {
    return "{" + cost + " cost, " + rowCount + " rows, " + cpu + " cpu, " + io + " io}";
  }

  /** Implementation of {@link org.apache.calcite.plan.RelOptCostFactory}
   * that creates {@link org.apache.calcite.plan.volcano.MyCost}s. */
  private static class Factory implements RelOptCostFactory {
    public RelOptCost makeCost(double dRows, double dCpu, double dIo) {
      return new MyCost(dRows, dCpu, dIo);
    }

    public RelOptCost makeHugeCost() {
      return MyCost.HUGE;
    }

    public RelOptCost makeInfiniteCost() {
      return MyCost.INFINITY;
    }

    public RelOptCost makeTinyCost() {
      return MyCost.TINY;
    }

    public RelOptCost makeZeroCost() {
      return MyCost.ZERO;
    }
  }
}
