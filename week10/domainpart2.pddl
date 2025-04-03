(define (domain taxi_part2)
  (:requirements :strips)

  (:types
    person - object
    location - object
    taxi - object
  )
  
  (:predicates
    (tlocation ?t - taxi ?l - location)
    (plocation ?p - person ?l - location)
    (connects ?from - location ?to - location)
    (outsidetaxi ?p - person)
    (insidetaxi ?p - person ?t - taxi)
    (empty ?t - taxi)
  )

  (:action get_in
    :parameters (?p - person ?t - taxi ?l - location)
    :precondition (and (plocation ?p ?l) (tlocation ?t ?l) (outsidetaxi ?p) (empty ?t))
    :effect (and (insidetaxi ?p ?t) (not (outsidetaxi ?p)) (not (empty ?t)))
  )

  (:action get_out
    :parameters (?p - person ?t - taxi ?l - location)
    :precondition (and (insidetaxi ?p ?t) (tlocation ?t ?l))
    :effect (and (plocation ?p ?l) (outsidetaxi ?p) (not (insidetaxi ?p ?t)) (empty ?t))
  )

  (:action move-taxi
    :parameters (?t - taxi ?from - location ?to - location)
    :precondition (and 
      (tlocation ?t ?from) 
      (connects ?from ?to)
      )
    :effect (and (tlocation ?t ?to) (not (tlocation ?t ?from)))
  )
  
)