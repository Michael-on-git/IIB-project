module chained_mux(
    input [3:0] a, b, c,
    input       sel1, sel2,
    output [3:0] y
);
    wire [3:0] mid;
    assign mid = sel1 ? b : a;
    assign y   = sel2 ? c : mid;
endmodule