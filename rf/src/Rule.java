public class Rule {

	private final String attribute;

	private final Predicate predicate;

	private final Object sampleValue;

	public Rule(String attribute, Predicate predicate, Object sampleValue) {
		this.attribute = attribute;
		this.predicate = predicate;
		this.sampleValue = sampleValue;
	}

	public boolean match(Item item) {
		Object otherField = item.getFieldValue(this.attribute);
		return this.predicate.eval(otherField, this.sampleValue);
	}

	public String getAttribute() {
		return this.attribute;
	}

	public Predicate getPredicate() {
		return this.predicate;
	}

	public Object getSampleValue() {
		return this.sampleValue;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result
				+ ((attribute == null) ? 0 : attribute.hashCode());
		result = prime * result
				+ ((predicate == null) ? 0 : predicate.hashCode());
		result = prime * result
				+ ((sampleValue == null) ? 0 : sampleValue.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Rule other = (Rule) obj;
		if (attribute == null) {
			if (other.attribute != null)
				return false;
		} else if (!attribute.equals(other.attribute))
			return false;
		if (predicate != other.predicate)
			return false;
		if (sampleValue == null) {
			if (other.sampleValue != null)
				return false;
		} else if (!sampleValue.equals(other.sampleValue))
			return false;
		return true;
	}
}
