use basilica_common::node_identity::NodeId;

fn main() {
    let seed = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: compute_node_id <seed>");
        std::process::exit(1);
    });
    let node_id = NodeId::new(&seed).expect("Failed to create NodeId");
    let bytes = node_id.uuid.into_bytes();
    println!("0x{}", hex::encode(bytes));
}
