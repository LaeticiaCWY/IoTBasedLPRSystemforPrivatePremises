// CombinedDisplay.js

import React from "react";
import ImageDisplay from "./ImageDisplay";
import TextDisplay from "./TextDisplay";

const CombinedDisplay = () => {
  return (
    <div>
      <ImageDisplay />
      <TextDisplay />
    </div>
  );
};

export default CombinedDisplay;
