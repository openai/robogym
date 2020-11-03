(import "base.libsonnet") + {
    geom+: {
        density: "1100.0",
        friction: "0.9 0.25 0.001",

        # TODO: we may want to reduce damping in solref here to make the ball bouncier.
    },
}
