(import "base.libsonnet") + {
    geom+: {
        density: "750.0",
        friction: "0.85 0.25 0.001",

        # A higher solimp makes it possible to grasp chess pieces.
        solimp: "0.99 0.999 0.001",
    },
}
