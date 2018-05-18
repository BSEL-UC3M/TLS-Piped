struct PotentialAndIndex {
  float Potential;
  unsigned short x;
  unsigned short y;
  unsigned short z;
  
  bool operator> (const PotentialAndIndex A)const
    {
    if(Potential>A.Potential)
      {
      return true;
      }
    else
      {
      return false;
      }
    }
    
  bool operator< (const PotentialAndIndex A)const
    {
    if(Potential<A.Potential)
      {
      return true;
      }
    else
      {
      return false;
      }
    }
};
